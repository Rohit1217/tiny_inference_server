import torch
import triton
import triton.testing
import triton.language as tl

def get_configs_flash_kernel():
    #TILE_M is in PAGES, so TILE_M*TOKENS_PER_BLOCK tokens per program.
    #small tiles matter for short contexts: with TILE_M=64 and kv_len=512 the whole
    #sequence is one tile, so only batch*num_head programs launch and the gpu idles.
    return [
        triton.Config({'TILE_M': 4}, num_warps=4),
        triton.Config({'TILE_M': 8}, num_warps=4),
        triton.Config({'TILE_M': 16}, num_warps=4),
        triton.Config({'TILE_M': 32}, num_warps=4),
        triton.Config({'TILE_M': 64}, num_warps=4),
        triton.Config({'TILE_M': 128}, num_warps=4),
        triton.Config({'TILE_M': 128}, num_warps=8),
        triton.Config({'TILE_M': 256}, num_warps=8),
    ]

def get_configs_reduce_kernel():
    return [
        triton.Config({'TILE_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'TILE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'TILE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'TILE_K': 128}, num_warps=8, num_stages=4),
    ]

@triton.autotune(
    configs=get_configs_flash_kernel(),
    key=['kv_len_bucket', 'HEAD_DIM'] 
)
@triton.jit
def paged_flash_decode_kernel(q_ptr,page_table_ptr,kv_base_ptr,lse_ptr,out_ptr,cuseq_ptr,kv_len_ptr,kv_len_bucket,num_groups:tl.constexpr,
                              stride_score_h:tl.constexpr,stride_lse_b:tl.constexpr,stride_lse_h:tl.constexpr,stride_score_b:tl.constexpr,stride_score_d:tl.constexpr,
                              page_tokens_stride:tl.constexpr,page_head_stride:tl.constexpr,page_kv_stride:tl.constexpr,page_block_stride:tl.constexpr,
                              sm_scale:tl.constexpr,TOKENS_PER_BLOCK:tl.constexpr,
                              HEAD_DIM:tl.constexpr,TILE_M:tl.constexpr,TILE_N:tl.constexpr):
    
    #page table be of form a contiguous shape with cuseq defining offsets of docs

    pid=tl.program_id(axis=0) #which block we are processning
    pid_h=tl.program_id(axis=1) #which head
    pid_b=tl.program_id(axis=2) #which user/batch

    page_b_start=tl.load(cuseq_ptr+pid_b)

    kv_len=tl.load(kv_len_ptr+pid_b)
    pid_kh=pid_h//num_groups

    if pid*TOKENS_PER_BLOCK*TILE_M>=kv_len:
        return

    num_heads=tl.num_programs(axis=1) 

    curr_page_ptr=page_table_ptr+page_b_start+pid*TILE_M
    #each program/block reads tile_m pages this is starting point for curr program

    #q_ptr load q as a block with padded zeros for tl dot
    q_blk_ptr=tl.make_block_ptr(q_ptr+pid_b*num_heads*HEAD_DIM,shape=(num_heads,HEAD_DIM),strides=(HEAD_DIM,1),
                                offsets=(pid_h,0),block_shape=(TILE_N,HEAD_DIM),order=(1,0))
    #out_ptr is of shape num_head,kv_len//TILE_M,HEAD_DIM
    q_mask=tl.arange(0,TILE_N)<1

    q=tl.load(q_blk_ptr,boundary_check=(0,),padding_option="zero")

    acc=tl.zeros((TILE_N,HEAD_DIM,),dtype=tl.float32)
    acc_max=float("-inf")
    acc_sum=0.0

    t=tl.arange(0,TOKENS_PER_BLOCK)
    d=tl.arange(0,HEAD_DIM)
    base_offset=pid_kh*page_head_stride 
    #page idx where we finish for current kv 

    total_pages=tl.cdiv(kv_len,TOKENS_PER_BLOCK)
    limit=tl.minimum(TILE_M,total_pages-pid*TILE_M)   #End of curr loop depending on  if we hit total pages

    for j in range(limit):
        page_idx=tl.load(curr_page_ptr + j)
        page_offset= page_idx*page_block_stride

        #read 2d kv from 2d ptrs made using strides and base ptr for a given page, 
        k=tl.load(kv_base_ptr+base_offset+page_offset+t[:,None]*page_tokens_stride+d[None,:])
        v=tl.load(kv_base_ptr+base_offset+page_offset+page_kv_stride+t[:,None]*page_tokens_stride+d[None,:])

        #reading 2d pointers with head dim
        scores=tl.dot(q,tl.trans(k))
        scores*=sm_scale #sm_scale will be scaled for 2 pow operation

        #algorithm same as flash atten we caclulate global lse at start using sum and max
        kv_offsets=(pid*TILE_M+j)*TOKENS_PER_BLOCK+tl.arange(0,TOKENS_PER_BLOCK)

        kv_mask=(kv_offsets<kv_len)
        mask=q_mask[:, None] & kv_mask[None, :]
        scores=tl.where(mask,scores,float("-inf"))

        new_max_val=tl.maximum(acc_max,tl.max(scores)) 
        scores_exp=tl.math.exp2(scores-new_max_val)

        update_factor=tl.math.exp2(acc_max-new_max_val)

        acc_sum=acc_sum*update_factor + tl.sum(scores_exp)
        acc_max=new_max_val
        acc=acc*update_factor

        #lse will be simply a float number max+log(sum e^xi-max)
        acc=tl.dot(scores_exp.to(tl.bfloat16),v,acc)
        
    lse=acc_max+tl.math.log2(acc_sum)
    acc=acc/acc_sum
    acc_row_first=tl.sum(acc,axis=0)
    
    out_offset=pid_b*stride_score_b + pid_h*stride_score_h + pid*stride_score_d + tl.arange(0, HEAD_DIM)
    tl.store(out_ptr+out_offset,acc_row_first)
    
    lse_offset=pid_b*stride_lse_b + pid_h*stride_lse_h + pid
    tl.store(lse_ptr+lse_offset,lse)


@triton.autotune(
    configs=get_configs_reduce_kernel(),
    key=['blocks_bucket', 'HEAD_DIM']
)
@triton.jit
def reduce_scores_kernel(score_ptr,lse_ptr,out_ptr,score_len_ptr,blocks_bucket,
                            stride_score_h, stride_score_b,stride_score_d, stride_lse_h,stride_lse_b,
                            stride_out_b,HEAD_DIM:tl.constexpr,TILE_K:tl.constexpr):

    pid_h=tl.program_id(axis=0)
    pid_b=tl.program_id(axis=1)

    score_len=tl.load(score_len_ptr+pid_b)


    # offset=pid_h*score_len*HEAD_DIM

    score_blk_ptr = tl.make_block_ptr(score_ptr + pid_b*stride_score_b + pid_h * stride_score_h, 
                                      shape=(score_len, HEAD_DIM), strides=(stride_score_d, 1),
                                      offsets=(0, 0), block_shape=(TILE_K, HEAD_DIM), order=(1, 0))
       
    base_lse_ptr = lse_ptr + pid_b*stride_lse_b+pid_h * stride_lse_h
       
    token_offsets=tl.arange(0,TILE_K)

    mask=token_offsets<score_len

    acc_lse=float("-inf")
    acc_score=tl.zeros((HEAD_DIM,),tl.float32)
    

    limit=(score_len//TILE_K)*TILE_K

    #non boundary computation
    for i in range(0,limit,TILE_K):
        #load lse of size TILE_K,
        lse_ptr_b=base_lse_ptr+i+tl.arange(0,TILE_K)
        lse_val=tl.load(lse_ptr_b)

        #calc new max lse
        new_max_lse=tl.maximum(tl.max(lse_val),acc_lse)

        score=tl.load(score_blk_ptr,boundary_check=(0,),padding_option="zero")

        #update lse_global which is given by lse_max+tl.log(e^(lse_i-lse_max)+e^(lse_j-lse_max))
        lse_updated= new_max_lse + tl.math.log2(tl.math.exp2(acc_lse-new_max_lse)+tl.sum(tl.math.exp2(lse_val-new_max_lse)))
        
        #update score and accumulate e^xi-lse_old*(lse_old-lse_updated)=e^xi-lse_updated

        update_factor=tl.math.exp2(lse_val-lse_updated) #size TILE_K,

        acc_score=tl.sum(score*update_factor[:,None],axis=0) + acc_score*tl.math.exp2(acc_lse-lse_updated)

        acc_lse=lse_updated
        score_blk_ptr=tl.advance(score_blk_ptr,(TILE_K,0))


    for i in range(limit,score_len,TILE_K):
        #load lse of size TILE_K,
        offset_idx=i+tl.arange(0,TILE_K)
        mask=offset_idx<score_len
        lse_ptr_b=base_lse_ptr+offset_idx

        lse_val=tl.load(lse_ptr_b,mask=mask,other=float("-inf"))

        #calc new max lse
        new_max_lse=tl.maximum(tl.max(lse_val),acc_lse)

        score=tl.load(score_blk_ptr,boundary_check=(0,),padding_option="zero")

        #update lse_global which is given by lse_max+tl.log(e^(lse_i-lse_max)+e^(lse_j-lse_max))
        lse_updated= new_max_lse + tl.math.log2(tl.math.exp2(acc_lse-new_max_lse)+tl.sum(tl.math.exp2(lse_val-new_max_lse)))
        
        #update score and accumulate e^xi-lse_old*(lse_old-lse_updated)=e^xi-lse_updated

        update_factor=tl.math.exp2(lse_val-lse_updated) #size TILE_K,

        acc_score=tl.sum(score*update_factor[:,None],axis=0) + acc_score*tl.math.exp2(acc_lse-lse_updated)

        acc_lse=lse_updated


    out_off=out_ptr+pid_b*stride_out_b+pid_h*HEAD_DIM +tl.arange(0,HEAD_DIM)
    tl.store(out_off,acc_score.to(tl.bfloat16))


def paged_flash_decode(q,page_table,block,kv_lens,num_groups,cuseq,max_seq_len,layer=0):
    #q is [batch,num_head,head_dim] one decode token per user
    #page_table is flat int32 of physical page ids, cuseq is cumulative PAGE counts
    #kv_lens is context length in TOKENS per user

    batch,num_head,head_dim=q.shape

    sm_scale=(1.44269504/(head_dim**0.5))
    tokens_per_block=block.tokens_per_block

    #layer is only a base ptr slice so kernel never sees a layer term
    kv_base=block.block[layer]

    #partials sized for smallest TILE_M since tile size known only after autotune
    max_pages=triton.cdiv(max_seq_len,tokens_per_block)
    MIN_TILE_M=min(c.kwargs["TILE_M"] for c in get_configs_flash_kernel())
    max_tiles=triton.cdiv(max_pages,MIN_TILE_M)

    #coarse key so autotune sweep does not rerun every step
    kv_len_bucket=triton.next_power_of_2(max_seq_len)

    #zeros/-inf so an unwritten partial contributes nothing in reduce, exp2(-inf-x)=0
    scores=torch.zeros((batch,num_head,max_tiles,head_dim),dtype=torch.float32,device=q.device)
    lse=torch.full((batch,num_head,max_tiles),float("-inf"),dtype=torch.float32,device=q.device)

    grid_flash=lambda META:(triton.cdiv(max_pages,META["TILE_M"]),num_head,batch)

    paged_flash_decode_kernel[grid_flash](
        q,page_table,kv_base,lse,scores,cuseq,kv_lens,kv_len_bucket,num_groups,
        scores.stride(1),lse.stride(0),lse.stride(1),scores.stride(0),scores.stride(2),
        block.tokens_stride,block.head_stride,block.kv_stride,block.block_stride,
        sm_scale,tokens_per_block,
        HEAD_DIM=head_dim,TILE_N=16
    )

    best_config=paged_flash_decode_kernel.best_config
    winning_tile_m=best_config.kwargs["TILE_M"]

    num_pages=(kv_lens+tokens_per_block-1)//tokens_per_block
    num_tiles=((num_pages+winning_tile_m-1)//winning_tile_m).to(torch.int32)

    blocks_bucket=triton.next_power_of_2(triton.cdiv(max_pages,winning_tile_m))

    out=torch.empty((batch,num_head,head_dim),dtype=q.dtype,device=q.device)

    grid_reduce=(num_head,batch)

    reduce_scores_kernel[grid_reduce](
        scores,lse,out,num_tiles,blocks_bucket,
        scores.stride(1),scores.stride(0),scores.stride(2),
        lse.stride(1),lse.stride(0),out.stride(0),
        HEAD_DIM=head_dim
    )

    return out


if __name__=="__main__":


    #BENCHMARKING CODE WRITTTEN BY CLAUDE

    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from block_table import block_table

    device="cuda"
    num_layers=2
    num_head=8
    head_dim=64
    tokens_per_block=16
    layer=1

    #ragged lengths so we exercise partial last page + multi tile split
    kv_lens_list=[37,512,2000,20000] #20000 forces >1 tile even for the largest TILE_M
    batch=len(kv_lens_list)

    free_memory=256*1024*1024
    pool=block_table(num_layers,num_head,head_dim,free_memory,tokens_per_block,device)
    pool.block.zero_() #torch.empty leaves garbage in the unused tail slots

    #shuffle the free list so pages come out scattered, not 0,1,2,...
    #otherwise a kernel that wrongly assumed contiguous pages would still pass
    import random
    from collections import deque
    random.seed(0)
    shuffled=list(pool.free_block_queue)
    random.shuffle(shuffled)
    pool.free_block_queue=deque(shuffled)

    print(f"pool: {pool.num_blocks} blocks of {tokens_per_block} tokens")

    page_ids_all=[]
    cuseq_list=[0]
    ref_k=[]
    ref_v=[]

    for kv_len in kv_lens_list:
        npages=triton.cdiv(kv_len,tokens_per_block)
        ids=[pool.allocate_block() for _ in range(npages)]
        page_ids_all+=ids
        cuseq_list.append(cuseq_list[-1]+npages)

        k=torch.randn(num_head,kv_len,head_dim,dtype=torch.bfloat16,device=device)
        v=torch.randn(num_head,kv_len,head_dim,dtype=torch.bfloat16,device=device)
        ref_k.append(k)
        ref_v.append(v)

        #scatter this user's kv into its physical pages, one page at a time
        for p,pg in enumerate(ids):
            s=p*tokens_per_block
            e=min(s+tokens_per_block,kv_len)
            n=e-s
            pool.block[layer,:,0,pg,:n,:]=k[:,s:e,:]
            pool.block[layer,:,1,pg,:n,:]=v[:,s:e,:]

    q=torch.randn(batch,num_head,head_dim,dtype=torch.bfloat16,device=device)
    page_table=torch.tensor(page_ids_all,dtype=torch.int32,device=device)
    cuseq=torch.tensor(cuseq_list,dtype=torch.int32,device=device)
    kv_lens=torch.tensor(kv_lens_list,dtype=torch.int32,device=device)

    max_seq_len=max(kv_lens_list) #host side, no device sync
    out=paged_flash_decode(q,page_table,pool,kv_lens,cuseq,max_seq_len,layer=layer)

    #reference: cudnn sdpa per user, 4d so the fused decode kernel is eligible
    ref=torch.empty_like(out)
    for i,kv_len in enumerate(kv_lens_list):
        q4=q[i][None,:,None,:]
        k4=ref_k[i][None]
        v4=ref_v[i][None]
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o=F.scaled_dot_product_attention(q4,k4,v4)
        ref[i]=o[0,:,0,:]

    print("-"*60)
    print(f"{'kv_len':<10} | {'max diff':<12} | {'close?':<8}")
    print("-"*60)
    for i,kv_len in enumerate(kv_lens_list):
        d=(ref[i]-out[i]).abs().max().item()
        ok=torch.allclose(ref[i],out[i],rtol=1e-2,atol=1e-2)
        print(f"{kv_len:<10} | {d:<12.5f} | {str(ok):<8}")
    print("-"*60)
    print("ALL CLOSE:",torch.allclose(ref,out,rtol=1e-2,atol=1e-2))

    #------------------------------------------------------------------
    # speed
    #------------------------------------------------------------------

    def build_pool(kv_lens_list,num_layers,num_head,head_dim,tokens_per_block,layer,free_memory,device):
        #fresh pool + pages + scattered kv, returns everything the kernel needs
        pool=block_table(num_layers,num_head,head_dim,free_memory,tokens_per_block,device)
        pool.block.zero_()

        random.seed(0)
        shuffled=list(pool.free_block_queue)
        random.shuffle(shuffled)
        pool.free_block_queue=deque(shuffled)

        page_ids_all=[]
        cuseq_list=[0]
        ks=[]
        vs=[]

        for kv_len in kv_lens_list:
            npages=triton.cdiv(kv_len,tokens_per_block)
            ids=[pool.allocate_block() for _ in range(npages)]
            page_ids_all+=ids
            cuseq_list.append(cuseq_list[-1]+npages)

            k=torch.randn(num_head,kv_len,head_dim,dtype=torch.bfloat16,device=device)
            v=torch.randn(num_head,kv_len,head_dim,dtype=torch.bfloat16,device=device)
            ks.append(k)
            vs.append(v)

            for p,pg in enumerate(ids):
                s=p*tokens_per_block
                e=min(s+tokens_per_block,kv_len)
                n=e-s
                pool.block[layer,:,0,pg,:n,:]=k[:,s:e,:]
                pool.block[layer,:,1,pg,:n,:]=v[:,s:e,:]

        return (pool,
                torch.tensor(page_ids_all,dtype=torch.int32,device=device),
                torch.tensor(cuseq_list,dtype=torch.int32,device=device),
                torch.tensor(kv_lens_list,dtype=torch.int32,device=device),
                ks,vs)


    bench_layers=1 #keep the pool small so long contexts fit
    bench_head=8
    bench_hd=64
    bench_tpb=16
    bench_layer=0
    bench_batch=4
    bench_mem=512*1024*1024

    print()
    print("="*84)
    print(f"UNIFORM BATCH (batch={bench_batch}) : paged, 1 launch   vs   cuDNN SDPA, 1 batched call")
    print("="*84)
    print(f"{'kv_len':<9}|{'paged (ms)':<13}|{'cuDNN (ms)':<13}|{'ratio':<9}|{'TILE_M':<8}|{'paged GB/s':<11}")
    print("-"*84)

    for L in [512,2048,8192,16384]:
        kvl=[L]*bench_batch
        pool_b,pt,cs,kl,ks,vs=build_pool(kvl,bench_layers,bench_head,bench_hd,
                                         bench_tpb,bench_layer,bench_mem,device)
        qb=torch.randn(bench_batch,bench_head,bench_hd,dtype=torch.bfloat16,device=device)
        msl=max(kvl)

        paged_flash_decode(qb,pt,pool_b,kl,cs,msl,layer=bench_layer) #warm + autotune

        k4=torch.stack(ks)
        v4=torch.stack(vs)
        q4=qb[:,:,None,:]

        def run_cudnn():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(q4,k4,v4)

        ms_p=triton.testing.do_bench(lambda:paged_flash_decode(qb,pt,pool_b,kl,cs,msl,layer=bench_layer),
                                     quantiles=[0.5])
        ms_c=triton.testing.do_bench(run_cudnn,quantiles=[0.5])

        tm=paged_flash_decode_kernel.best_config.kwargs["TILE_M"]
        kv_bytes=2*bench_batch*bench_head*L*bench_hd*2 #k+v, bf16
        gbs=kv_bytes/(ms_p*1e-3)/1e9

        print(f"{L:<9}|{ms_p:<13.4f}|{ms_c:<13.4f}|{ms_c/ms_p:<9.2f}|{tm:<8}|{gbs:<11.0f}")

        del pool_b,k4,v4
        torch.cuda.empty_cache()

    print()
    print("="*84)
    print("RAGGED BATCH : paged, 1 launch   vs   cuDNN, one call per user (what dense must do)")
    print("="*84)

    kvl=[128,900,4096,15000,300,7000,2048,11000]
    pool_b,pt,cs,kl,ks,vs=build_pool(kvl,bench_layers,bench_head,bench_hd,
                                     bench_tpb,bench_layer,bench_mem,device)
    qb=torch.randn(len(kvl),bench_head,bench_hd,dtype=torch.bfloat16,device=device)
    msl=max(kvl)

    paged_flash_decode(qb,pt,pool_b,kl,cs,msl,layer=bench_layer) #warm + autotune

    def run_cudnn_loop():
        outs=[]
        for i in range(len(kvl)):
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                outs.append(F.scaled_dot_product_attention(qb[i][None,:,None,:],ks[i][None],vs[i][None]))
        return outs

    ms_p=triton.testing.do_bench(lambda:paged_flash_decode(qb,pt,pool_b,kl,cs,msl,layer=bench_layer),
                                 quantiles=[0.5])
    ms_c=triton.testing.do_bench(run_cudnn_loop,quantiles=[0.5])

    print(f"lens                  : {kvl}")
    print(f"paged (1 launch)      : {ms_p:.4f} ms")
    print(f"cuDNN ({len(kvl)} launches)   : {ms_c:.4f} ms")
    print(f"speedup               : {ms_c/ms_p:.2f}x")
    print("="*84)
