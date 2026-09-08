import torch
from collections import deque
import triton
import triton.language as tl

@triton.jit
def write_kv_cache_decode(block_ptr_kv,page_idx,k_ptr,v_ptr,kv_stride:tl.constexpr,
                          head_stride:tl.constexpr,HEAD_DIM:tl.constexpr):
    #kv ptr size is B*L*N_H*H (batch,layers,num_head,head_dim). Total memory is around 32*batch KB for our setup
    pid_h=tl.program_id(axis=0)
    pid_b=tl.program_id(axis=1)
    # pid_t=tl.program_id(axis=0)

    num_heads=tl.num_programs(axis=0) #heads are on axis 0, same axis as pid_h

    curr_page_idx = tl.load(page_idx + pid_b)
    write_idx=curr_page_idx + pid_h*head_stride # k write ptr in block_table
    #curr_page_idx already holds block*block_stride+token_pos*tokens_stride from allocate_cache

    stride=pid_b*num_heads*HEAD_DIM+pid_h*HEAD_DIM+tl.arange(0,HEAD_DIM) #loading from torch output so of shape N,NH,H

    k=tl.load(k_ptr+stride)
    v=tl.load(v_ptr+stride)

    write_offset=write_idx+tl.arange(0,HEAD_DIM)
    
    tl.store(block_ptr_kv+write_offset,k)
    tl.store(block_ptr_kv+write_offset+kv_stride,v) # v write ptr using kv stride, shape is Head,D_model


class block_table:
    def __init__(self,num_layers,num_heads,head_dim,free_memory,tokens_per_block,device):
        #size of a single token cache across layers and kv combined
        one_kv_mem=2*2*num_heads*num_layers*head_dim

        #number of pages
        self.page_size=tokens_per_block*one_kv_mem
        self.num_blocks=free_memory//self.page_size

        #zeros not empty: unwritten tail slots feed tl.dot in the flash kernel, garbage inf/nan there gives 0*inf=nan
        self.block=torch.zeros((num_layers,num_heads,2,self.num_blocks,tokens_per_block,head_dim),dtype=torch.bfloat16,device=device)
        
        #block_idx queue for allocating and managing blocks
        self.free_block_queue=deque(list(range(self.num_blocks)))

        self.device=device

        self.num_heads=num_heads
        self.num_layers=num_layers
        self.head_dim=head_dim
        self.tokens_per_block=tokens_per_block

        # Shape of block table will be of form layers,heads,2(key,value)
        # assigning an page idx implicitly sets layer wise ,head wise and kv indices by offset
        self.layer_stride,self.head_stride,self.kv_stride,self.block_stride,self.tokens_stride,_=self.block.stride()
    
    def allocate_block(self):
        return self.free_block_queue.popleft()
    
    
    def free_block(self,block_idx):
        self.free_block_queue.append(block_idx)
        return


class User:
    def __init__(self,global_block_table:block_table):
        self.block_table=global_block_table
        self.block_ids=[]

        self.block_stride=self.block_table.block_stride
        self.tokens_stride=self.block_table.tokens_stride
        self.tokens_per_block=self.block_table.tokens_per_block

    def slot(self,pos):
        #write ptr for token position pos, allocating blocks as needed. position-addressed so a
        #rolled-back slot (rejected draft kv) is simply overwritten when the position is re-fed
        while len(self.block_ids)<=pos//self.tokens_per_block:
            self.block_ids.append(self.block_table.allocate_block())
        return self.block_ids[pos//self.tokens_per_block]*self.block_stride+(pos%self.tokens_per_block)*self.tokens_stride

    def free_cache(self):
        for block_id in self.block_ids:
            self.block_table.free_block(block_id)
        
        return


