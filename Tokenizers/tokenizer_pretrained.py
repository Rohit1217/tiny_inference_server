import tiktoken
import pandas as pd
import os
import numpy as np
import gc
from  tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")
text="hello world"
ids = enc.encode_ordinary(text)
  # ordinary = no special token handling, faster


df=pd.read_parquet("../data/fineweb-edu-10BT/sample/10BT/000_00000.parquet")
texts=df['text'].tolist()

chunk_length=len(texts)//4

chunks=[texts[idx*chunk_length:(idx+1)*chunk_length] for idx in range(4)]

def write_chunk(arr,path):
    tmp = path + ".tmp"
    arr.tofile(tmp)
    os.rename(tmp,path)


def tokenize(text,enc):
    encoded_ids=enc.encode_ordinary_batch(text,num_threads=24)
    for ids in encoded_ids:
        ids.append(enc.eot_token)    
    token_arr=np.concatenate(encoded_ids,dtype=np.uint16,casting="unsafe")
    return token_arr


i=0
for j in tqdm(range(14)):
    if j<10:
        df=pd.read_parquet(f"../data/fineweb-edu-10BT/sample/10BT/00{j}_00000.parquet")
    else:
        df=pd.read_parquet(f"../data/fineweb-edu-10BT/sample/10BT/0{j}_00000.parquet")
    
    texts=df['text'].tolist()
    chunk_length=len(texts)//4
    chunks=[texts[idx*chunk_length:(idx+1)*chunk_length] for idx in range(4)]
    del df
    del texts
    gc.collect()

    for chunk in tqdm(chunks):
        token_arr=tokenize(chunk,enc)
        write_chunk(token_arr,f"/home/rohit1/data/fineweb-edu-10BT/sample/10BT/chunk_{i}.bin")
        i+=1
        del token_arr
        gc.collect()

    