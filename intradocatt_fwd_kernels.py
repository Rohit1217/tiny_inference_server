#Varlen causal flash attention (causal within each doc, cuseq token boundaries).
#Copied from LLM_train/Intradoc_kernels (forward only) for batched prefill.
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
    ],
    key=[ 'NUM_HEADS','HEAD_DIM','group_size'],
)
@triton.jit
def flash_att_intra_doc_masked(q_ptr,k_ptr,v_ptr,o_ptr,cuseq_ptr,lse_ptr,total_tokens,sm_scale:tl.constexpr,
                               group_size:tl.constexpr,HEAD_DIM:tl.constexpr,
                                BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr):

    #Read grid
    pid=tl.program_id(axis=0)
    pid_doc=tl.program_id(axis=1)
    pid_h=tl.program_id(axis=2)

    #Assume data in shape (num_tokens,num_head,head_dim)
    #Load the doc start and end idx and calculate doc length using them

    start=tl.load(cuseq_ptr+pid_doc)
    end=tl.load(cuseq_ptr+pid_doc+1)

    doc_len=end-start

    if pid*BLOCK_M >= doc_len:
        return

    #Load q,k,v ptrs
    stride_tok_q=tl.num_programs(axis=2) * HEAD_DIM
    stride_tok_k=stride_tok_q//group_size
    pid_kh=pid_h//group_size

    q_ptr_b=q_ptr+start*stride_tok_q+ pid_h*HEAD_DIM
    k_ptr_b=k_ptr+start*stride_tok_k + pid_kh*HEAD_DIM
    v_ptr_b=v_ptr+start*stride_tok_k + pid_kh*HEAD_DIM
    o_ptr_b=o_ptr+start*stride_tok_q+ pid_h*HEAD_DIM

    lse_ptr_b=lse_ptr+pid_h*total_tokens + start

    q_block_ptr=tl.make_block_ptr(base=q_ptr_b,
                                  shape=(doc_len,HEAD_DIM),
                                  strides=(stride_tok_q,1),
                                  offsets=(pid*BLOCK_M,0),
                                  block_shape=(BLOCK_M,HEAD_DIM),
                                  order=(1,0))

    #Load k transposed,now changed since tl.trans(k) inside tl.dot is faster
    k_block_ptr=tl.make_block_ptr(k_ptr_b,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                                       offsets=(0,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))

    v_block_ptr=tl.make_block_ptr(base=v_ptr_b,
                                shape=(doc_len,HEAD_DIM),
                                strides=(stride_tok_k,1),
                                offsets=(0,0),
                                block_shape=(BLOCK_N,HEAD_DIM),
                                order=(1,0))

    o_block_ptr=tl.make_block_ptr(base=o_ptr_b,
                                  shape=(doc_len,HEAD_DIM),
                                  strides=(stride_tok_q,1),
                                  offsets=(pid*BLOCK_M,0),
                                  block_shape=(BLOCK_M,HEAD_DIM),
                                  order=(1,0))

    lse_block_ptr=tl.make_block_ptr(base=lse_ptr_b,shape=(doc_len,),
                            strides=(1,),offsets=(pid*BLOCK_M,),
                            block_shape=(BLOCK_M,),order=(0,))

    #scale trick: fold 1/ln(2) into qk so we can use exp2 (hardware-accelerated) instead of exp
    qk_scale = sm_scale * 1.44269504

    q = tl.load(q_block_ptr,boundary_check=(0,),padding_option="zero")

    #Buffers to accumulate res ,max and sums
    acc=tl.zeros((BLOCK_M,HEAD_DIM),tl.float32)
    max_val=tl.full((BLOCK_M,),float("-inf"),tl.float32)
    sum_val=tl.zeros((BLOCK_M,),tl.float32)

    q_offset=pid * BLOCK_M + tl.arange(0, BLOCK_M)
    doc_bias=tl.where(q_offset<doc_len,0.0,-10000.0) # offsets for  doc masking which q crossing boundary


    #safe part no causality got till start of query block
    for i in range(0,pid*BLOCK_M,BLOCK_N): #There will be some tail  q # We can assume that q block size will be greater than kv

        k=tl.load(k_block_ptr,boundary_check=(0,),padding_option="zero")
        v=tl.load(v_block_ptr,boundary_check=(0,),padding_option="zero")

        scores=tl.dot(q,tl.trans(k))
        scores=scores*qk_scale
        scores+=doc_bias[:,None]

        new_max=tl.maximum(max_val,tl.max(scores,axis=1))
        scores_exp=tl.math.exp2(scores-new_max[:,None])

        update_factor=tl.math.exp2(max_val-new_max)

        acc=acc*update_factor[:,None]

        acc=tl.dot(scores_exp.to(tl.bfloat16),v,acc)

        sum_val=sum_val*update_factor + tl.sum(scores_exp,axis=1)
        max_val=new_max

        k_block_ptr=tl.advance(k_block_ptr,(BLOCK_N,0))
        v_block_ptr=tl.advance(v_block_ptr,(BLOCK_N,0))


    #Causal and doc masking(Need to do block masking padding for q,k,v)

    limit_causal = tl.minimum((pid + 1) * BLOCK_M, doc_len)
    for i in range(pid*BLOCK_M,limit_causal,BLOCK_N): #We ensure that BLOCK_N divides BLOCK_M so no tailwind,(Offcourse doc masking remains)
        k=tl.load(k_block_ptr,boundary_check=(0,),padding_option="zero")
        v=tl.load(v_block_ptr,boundary_check=(0,),padding_option="zero")

        scores=tl.dot(q,tl.trans(k))
        scores=scores*qk_scale

        k_offset=i+tl.arange(0,BLOCK_N)

        causal_bias=tl.where(q_offset[:, None] >= k_offset[None, :], 0.0, -10000.0)
        #This creates an M,N causal mask similar to float inf mask in standard torch inf
        scores=scores + causal_bias + doc_bias[:,None]

        new_max=tl.maximum(max_val,tl.max(scores,axis=1))
        scores_exp=tl.math.exp2(scores-new_max[:,None])
        update_factor=tl.math.exp2(max_val-new_max)

        acc=acc*update_factor[:,None]
        acc=tl.dot(scores_exp.to(tl.bfloat16),v,acc)

        sum_val=sum_val*update_factor + tl.sum(scores_exp,axis=1)
        max_val=new_max

        k_block_ptr=tl.advance(k_block_ptr,(BLOCK_N,0))
        v_block_ptr=tl.advance(v_block_ptr,(BLOCK_N,0))

    acc=acc/sum_val[:,None]
    lse=max_val + tl.math.log2(sum_val)

    tl.store(o_block_ptr, acc.to(tl.bfloat16), boundary_check=(0,))
    tl.store(lse_block_ptr, lse, boundary_check=(0,))

    return


def triton_intradoc_attention(q,k,v,cuseq,max_len,group_size):
    out=torch.empty_like(q)

    num_docs=cuseq.shape[0]-1
    total_tokens,num_heads,head_dim=q.shape
    lse=torch.empty((num_heads,total_tokens),device=q.device) #num heads,num_tokens for better coalesced mem access

    sm_scale=1/(head_dim**0.5)

    grid=lambda META: (triton.cdiv(max_len,META["BLOCK_M"]),num_docs,num_heads)
    flash_att_intra_doc_masked[grid](q,k,v,out,cuseq,lse,total_tokens,sm_scale,group_size,head_dim)
    return out,lse
