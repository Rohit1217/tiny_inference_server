import torch,triton,random
from collections import deque
from block_table import block_table
from paged_attention_kernels import paged_flash_decode
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

#GQA paged decode correctness: 8 query heads share 2 kv heads (group_size=4).
#pool is built with num_kv heads only, kernel maps q head -> kv head with num_groups=group_size.

dev="cuda";nl=1;num_q=8;num_kv=2;group_size=num_q//num_kv;hd=64;tpb=16;layer=0

pool=block_table(nl,num_kv,hd,128*1024*1024,tpb,dev) #POOL SIZED BY num_kv, group-agnostic
pool.block.zero_()
random.seed(0);s=list(pool.free_block_queue);random.shuffle(s);pool.free_block_queue=deque(s)

kvl=[37,512,2000];B=len(kvl)
pids=[];cus=[0];ks=[];vs=[]
for L in kvl:
    n=triton.cdiv(L,tpb);ids=[pool.allocate_block() for _ in range(n)]
    pids+=ids;cus.append(cus[-1]+n)
    k=torch.randn(num_kv,L,hd,dtype=torch.bfloat16,device=dev) #KV has num_kv heads
    v=torch.randn(num_kv,L,hd,dtype=torch.bfloat16,device=dev)
    ks.append(k);vs.append(v)
    for p,pg in enumerate(ids):
        a=p*tpb;b=min(a+tpb,L);m=b-a
        pool.block[layer,:,0,pg,:m,:]=k[:,a:b,:]
        pool.block[layer,:,1,pg,:m,:]=v[:,a:b,:]

pt=torch.tensor(pids,dtype=torch.int32,device=dev)
cs=torch.tensor(cus,dtype=torch.int32,device=dev)
kl=torch.tensor(kvl,dtype=torch.int32,device=dev)
q=torch.randn(B,num_q,hd,dtype=torch.bfloat16,device=dev) #Q has num_q heads

out=paged_flash_decode(q,pt,pool,kl,group_size,cs,max(kvl),layer)

#reference: expand kv heads to query heads, kv_head=qhead//group_size (repeat_interleave)
ref=torch.empty_like(out)
for i,L in enumerate(kvl):
    kq=ks[i].repeat_interleave(group_size,dim=0)
    vq=vs[i].repeat_interleave(group_size,dim=0)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        o=F.scaled_dot_product_attention(q[i][None,:,None,:],kq[None],vq[None])
    ref[i]=o[0,:,0,:]

print(f"num_q={num_q} num_kv={num_kv} group_size={group_size}")
print("GQA max diff:",round((ref-out).abs().max().item(),5),
      " close:",torch.allclose(ref,out,rtol=1e-2,atol=1e-2))

#negative control: passing num_kv (literal #groups) must NOT match
out_bad=paged_flash_decode(q,pt,pool,kl,num_kv,cs,max(kvl),layer)
print("num_groups=num_kv (wrong value) close:",
      torch.allclose(ref,out_bad,rtol=1e-2,atol=1e-2)," (expect False)")