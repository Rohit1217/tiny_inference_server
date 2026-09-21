import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo ROOT (parent of Generate/) -> Config/, Tokenizers/, main_models_train

import torch
import torch.nn.functional as F

from Config.config import Config
from Tokenizers.tokenizer_fast import Tokenizer

from main_models_new import Transformer

cfg=Config()

def load_tokenizer(tokenizer_path,vocab_size):
    tokenizer=Tokenizer(vocab_size=vocab_size)
    tokenizer.load_tokenizer(tokenizer_path)

    return tokenizer

@torch.no_grad
def load_ckpt(ckpt_path,device,test=False):
    model=Transformer(vocab_size=cfg.VOCAB_SIZE,max_context=cfg.MAX_CONTEXT,
                                max_freq=cfg.MAX_FREQ,d_model=cfg.D_MODEL,n_heads=cfg.N_HEAD,
                                num_layers=cfg.NUM_LAYERS,attn_dropout=cfg.ATT_DROPOUT,
                                ffn_hidden_dim=cfg.FFN_HIDDEN_DIM,ffn_dropout=cfg.FFN_DROPOUT,mtp_heads=cfg.MTP_HEADS,
                                grad_checkpoint_every=cfg.GRAD_CHECKPOINT_EVERY,group_size=cfg.NUM_GROUPS,max_len=cfg.MAX_CONTEXT,
                                intradoc_att=cfg.INTRADOC_ATT)
    
    if test:
        model=model.to(device).to(torch.bfloat16)
        model.buffers_to_float()
        model.eval()
        return model,"test"

    ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)

    p2=[p for n, p in model.named_parameters() if p.ndim == 2 and "embed" not in n]
    p1=[p for n, p in model.named_parameters() if p.ndim == 1 or "embed" in n]
    
    
    torch._foreach_copy_(p2,ckpt["master_2d"])
    torch._foreach_copy_(p1, ckpt["master_1d"])

    model=model.to(device).to(torch.bfloat16)
    model.buffers_to_float()
    model.eval()
    
    return model,int(ckpt.get("opt_step", 0))


@torch.no_grad
def generate_logits(ids,model,device):
    #some patchwork since we are using training forward method for this
    seq_len=len(ids)
    pad_seq_len=seq_len+cfg.MTP_HEADS+1

    token_pos=torch.arange(0,seq_len,device=device).long()
    cuseq=torch.tensor([0,seq_len],device=device,dtype=torch.int32)

    x=torch.tensor(ids+[48000]*(cfg.MTP_HEADS+1),device=device).long()
    x=x.unsqueeze(0)

    out=model(x,cuseq,token_pos)
    out=out[0,seq_len-1,:]
    logits=out.float()@model.embedding.weight.float().t()

    return logits

def sample(logits,temprature,top_k=0,top_p=0):
    logits=(logits/temprature)
    hidden_dim=logits.shape[-1]

    prob=F.softmax(logits,dim=-1)

    if top_k>0:
        k=torch.topk(-logits,hidden_dim-top_k)
        logits[k[1]]=float("-inf")
        
    if top_p>0:
        sorted_logits,sorted_idx=torch.sort(logits,descending=True)
        cum_prob=torch.cumsum(F.softmax(sorted_logits,dim=-1),dim=-1)
        remove=cum_prob>top_p


        remove[1:]=remove[:-1].clone()
        remove[0]=False        
        sorted_logits[remove]=float("-inf")
        logits=torch.full_like(logits, float("-inf")).scatter_(0, sorted_idx, sorted_logits)


    prob=F.softmax(logits,dim=-1)
    sample=int(torch.multinomial(prob,1))
    return sample

def generate(model,tokenizer,prompt,temprature,top_k,top_p,max_seq,device):
    ids=tokenizer.encode(prompt)
    start=len(ids)


    while len(ids)<max_seq:
        logits=generate_logits(ids,model,device)
        out_token=sample(logits,temprature,top_k,top_p)
        ids.append(out_token)

        if out_token == 48000:
            break
    return tokenizer.decode(ids[start:])


model,opt_step=load_ckpt(ckpt_path="Weights/checkpoints/Run_overfit-4-mtp_mp_gqa_ddp_intradoc_main_neww/latest.pt",
                device="cuda:0",test=False)

print("MODEL LOADED")
tokenizer=load_tokenizer(tokenizer_path="/home/rohit1/LLM_train/Tokenizers/tokenizer_vocab.titoken",vocab_size=48000)

print("TOKENIZER_LOADED")

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Hey, what are you doing?",
    "In 1969, humans first landed on the",
    "Water is made of hydrogen and",
    "Once upon a time, there was a small robot who",
    "The three primary colors are",
    "Question: What is the largest planet in our solar system?\nAnswer:",
    "def add(a, b):\n    return",
    "My favorite thing about winter is",
    "How to stump a model?"
]

for prompt in DEFAULT_PROMPTS:
    output=generate(model,tokenizer,prompt=prompt,temprature=0.9,top_k=50,top_p=0.95,max_seq=1024,device="cuda:0")

    print("="*100)
    print("PROMPT")
    print(prompt)
    print("OUTPUT")
    print(output)



