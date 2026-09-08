import torch
import torch.nn as nn
import torch.nn.functional as F

from paged_attention_kernels import paged_flash_decode
from intradocatt_fwd_kernels import triton_intradoc_attention
from block_table import write_kv_cache_decode

#NOTE: module registration order below must match the training model (LLM_train/model.py)
#so the flat master_2d/master_1d checkpoint lists line up. Trunk params are an exact
#prefix of those lists (mtp heads register after rms_out), so we load by prefix copy.

class linear_swig(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        std=torch.sqrt(torch.tensor(1/in_dim))
        self.weight=nn.Parameter(torch.randn(in_dim,out_dim)*std)

    def forward(self,x):
        return x@self.weight

class linear_proj(nn.Module):
    def __init__(self,in_dim,out_dim,num_layers):
        super().__init__()
        std=torch.sqrt(torch.tensor(1/(in_dim*2*num_layers)))
        self.weight=nn.Parameter(torch.randn(in_dim,out_dim)*std)

    def forward(self,x):
        return x@self.weight

class rms_norm(nn.Module):
    def __init__(self,num_dim):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(num_dim))
        self.register_buffer("rootn",torch.pow(torch.tensor(num_dim),0.5))
        self.register_buffer("eps",torch.tensor(2e-08))

    def forward(self,x):
        xdtype=x.dtype
        x=x.float()
        norm_rms=torch.norm(x,dim=-1,keepdim=True)
        return ((x*self.rootn)/(norm_rms+self.eps)*self.scale).to(dtype=xdtype)

class swiglu(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.linear1=linear_swig(in_dim,out_dim)
        self.linear2=linear_swig(in_dim,out_dim)

    def forward(self,x):
        return F.silu(self.linear1(x))*self.linear2(x)

class ffn_router(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_layers):
        super().__init__()
        self.swig=swiglu(in_dim,hidden_dim)
        self.fc=linear_proj(hidden_dim,in_dim,num_layers)

    def forward(self,x):
        return self.fc(self.swig(x))

#ROPE
def precompute_rope_fast(max_context,max_freq,head_dim):
    seq_pos_tensor=torch.arange(max_context)
    i=torch.arange(0,head_dim//2)

    theta=1/torch.pow(max_freq,(2*i.float())/head_dim)
    seq_pos_theta=torch.outer(seq_pos_tensor,theta)
    return torch.cos(seq_pos_theta).repeat(1,2),torch.sin(seq_pos_theta).repeat(1,2)

def rotate_half(x):
    x1,x2=x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)

def apply_rope_decode(q,k,cos_embed,sin_embed,positions):
    #positions (B,) long: rope index of the token being decoded. (B,1,hd) broadcasts over heads
    cos,sin=cos_embed[positions].unsqueeze(1),sin_embed[positions].unsqueeze(1)
    q_type=q.dtype
    qf,kf=q.float(),k.float()
    return (qf*cos+rotate_half(qf)*sin).to(q_type),(kf*cos+rotate_half(kf)*sin).to(q_type)


class paged_gqa(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,group_size):
        super().__init__()
        self.q_proj=linear_swig(d_model,d_model)
        self.kv_proj=linear_swig(d_model,(2*d_model)//group_size)
        self.rms_norm_att=rms_norm(d_model)
        self.linear_proj=linear_proj(d_model,d_model,num_layers)

        self.head_dim=d_model//n_heads
        self.kvn_heads=n_heads//group_size
        self.group_size=group_size
        self.n_heads=n_heads

    def decode(self,x,cos,sin,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len,layer):
        #x (B,D): B independent token rows. rows of one request share a page_start and differ in
        #position/kv_len, so a multi-row chunk (prefill, verify) is causal for free: all kv is
        #written first, then row at position p attends kv_len=p+1 tokens
        B,D=x.shape
        residual=x
        x=self.rms_norm_att(x)

        q=self.q_proj(x).view(B,self.n_heads,self.head_dim)
        kv=self.kv_proj(x).view(B,2,self.kvn_heads,self.head_dim)
        k,v=kv.unbind(1)

        q,k=apply_rope_decode(q,k,cos,sin,positions)
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()

        write_kv_cache_decode[(self.kvn_heads,B)](pool.block[layer],write_ptrs,k,v,
                                                  pool.kv_stride,pool.head_stride,HEAD_DIM=self.head_dim)

        out=paged_flash_decode(q,page_table,pool,kv_lens,self.group_size,page_starts,max_seq_len,layer)
        return residual+self.linear_proj(out.view(B,D))

    def prefill(self,x,cos,sin,pool,positions,write_ptrs,cuseq,max_len,layer):
        #x (T,D): concatenated request prompts, cuseq = cumulative TOKEN counts. valid only when
        #each request starts from an empty cache (positions restart at 0 per request), so the
        #dense k,v computed here IS the whole context and intradoc attention is exact
        B,D=x.shape
        residual=x
        x=self.rms_norm_att(x)

        q=self.q_proj(x).view(B,self.n_heads,self.head_dim)
        kv=self.kv_proj(x).view(B,2,self.kvn_heads,self.head_dim)
        k,v=kv.unbind(1)

        q,k=apply_rope_decode(q,k,cos,sin,positions)
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()

        write_kv_cache_decode[(self.kvn_heads,B)](pool.block[layer],write_ptrs,k,v,
                                                  pool.kv_stride,pool.head_stride,HEAD_DIM=self.head_dim)

        out,_=triton_intradoc_attention(q,k,v,cuseq,max_len,self.group_size)
        return residual+self.linear_proj(out.view(B,D))


class Transformer_block(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,ffn_hidden_dim,group_size):
        super().__init__()
        self.att=paged_gqa(d_model,n_heads,num_layers,group_size)
        self.ffn=ffn_router(d_model,ffn_hidden_dim,num_layers)
        self.rms_norm_ffn=rms_norm(d_model)

    def decode(self,x,cos,sin,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len,layer):
        x=self.att.decode(x,cos,sin,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len,layer)
        return x+self.ffn(self.rms_norm_ffn(x))

    def prefill(self,x,cos,sin,pool,positions,write_ptrs,cuseq,max_len,layer):
        x=self.att.prefill(x,cos,sin,pool,positions,write_ptrs,cuseq,max_len,layer)
        return x+self.ffn(self.rms_norm_ffn(x))


class mtp_head(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,ffn_hidden_dim,group_size):
        super().__init__()
        self.proj=linear_proj(2*d_model,d_model,num_layers=1)
        self.rms_embed=rms_norm(d_model)
        self.rms_out=rms_norm(d_model)
        self.trans_block=Transformer_block(d_model,n_heads,num_layers,ffn_hidden_dim,group_size)

    def decode(self,h,embed,cos,sin,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len,layer):
        x=self.proj(torch.cat([h,self.rms_embed(embed)],dim=-1))
        x=self.trans_block.decode(x,cos,sin,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len,layer)
        return self.rms_out(x)

    def prefill(self,h,embed,cos,sin,pool,positions,write_ptrs,cuseq,max_len,layer):
        x=self.proj(torch.cat([h,self.rms_embed(embed)],dim=-1))
        x=self.trans_block.prefill(x,cos,sin,pool,positions,write_ptrs,cuseq,max_len,layer)
        return self.rms_out(x)


class Transformer(nn.Module):
    def __init__(self,vocab_size,max_context,max_freq,d_model,n_heads,num_layers,ffn_hidden_dim,group_size,mtp_heads=0):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        nn.init.normal_(self.embedding.weight,mean=0.0,std=d_model**(-0.5))

        self.transformer_block_list=nn.ModuleList([Transformer_block(d_model,n_heads,num_layers,
                                                   ffn_hidden_dim,group_size) for _ in range(num_layers)])
        self.rms_out=rms_norm(d_model)
        self.mtp_heads_list=nn.ModuleList([mtp_head(d_model,n_heads,num_layers,ffn_hidden_dim,
                                           group_size) for _ in range(mtp_heads)])

        self.num_layers=num_layers
        self.num_mtp_heads=mtp_heads
        self.nkv_heads=n_heads//group_size
        self.head_dim=d_model//n_heads
        self.max_context=max_context

        cos,sin=precompute_rope_fast(max_context,max_freq,self.head_dim)
        self.register_buffer("cos",cos)
        self.register_buffer("sin",sin)

    @torch.no_grad()
    def trunk_rows(self,tokens,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len):
        #tokens (B,) independent rows -> rms_out hidden (B,D)
        x=self.embedding(tokens)
        for i,blk in enumerate(self.transformer_block_list):
            x=blk.decode(x,self.cos,self.sin,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len,i)
        return self.rms_out(x)

    @torch.no_grad()
    def mtp_rows(self,i,h,tokens,pool,page_table,page_starts,positions,kv_lens,write_ptrs,max_seq_len):
        #h (B,D) hiddens from the previous stage, tokens (B,) embed input. kv lives at layer num_layers+i
        embed=self.embedding(tokens)
        return self.mtp_heads_list[i].decode(h,embed,self.cos,self.sin,pool,page_table,page_starts,
                                             positions,kv_lens,write_ptrs,max_seq_len,self.num_layers+i)

    @torch.no_grad()
    def prefill_rows(self,tokens,pool,positions,write_ptrs,cuseq,max_len):
        #tokens (T,) concatenated prompts -> rms_out hidden (T,D). intradoc attention per layer
        x=self.embedding(tokens)
        for i,blk in enumerate(self.transformer_block_list):
            x=blk.prefill(x,self.cos,self.sin,pool,positions,write_ptrs,cuseq,max_len,i)
        return self.rms_out(x)

    @torch.no_grad()
    def mtp_prefill_rows(self,i,h,tokens,pool,positions,write_ptrs,cuseq,max_len):
        embed=self.embedding(tokens)
        return self.mtp_heads_list[i].prefill(h,embed,self.cos,self.sin,pool,positions,
                                              write_ptrs,cuseq,max_len,self.num_layers+i)

    def logits(self,h):
        return h@self.embedding.weight.T

    def buffers_to_float(self):
        for name,buffer in list(self.named_buffers()):
            if torch.is_floating_point(buffer):
                parent=self.get_submodule(name.rsplit(".",1)[0]) if "." in name else self
                setattr(parent,name.rsplit(".",1)[-1],buffer.float())

    @torch.no_grad()
    def load_ckpt(self,ckpt_path,device):
        ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)

        p2=[p for n,p in self.named_parameters() if p.ndim==2 and "embed" not in n]
        p1=[p for n,p in self.named_parameters() if p.ndim==1 or "embed" in n]

        #registration order matches training, so the flat lists line up. with fewer mtp heads
        #than training our params are still a prefix (mtp heads register after rms_out)
        torch._foreach_copy_(p2,ckpt["master_2d"][:len(p2)])
        torch._foreach_copy_(p1,ckpt["master_1d"][:len(p1)])

        self.to(device).to(torch.bfloat16)
        self.buffers_to_float()
        self.eval()
        return int(ckpt.get("opt_step",0))
