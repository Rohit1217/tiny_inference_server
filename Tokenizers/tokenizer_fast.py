import pandas as pd
import os
from collections import defaultdict,Counter
import regex as re
import heapq
from tqdm import tqdm
import tiktoken
import base64,pickle

# df=pd.read_parquet("../data/fineweb-edu-10BT/sample/10BT/000_00000.parquet")
# texts=df['text'].tolist()

GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
O200_K_PATTERN=r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}|_[\p{L}\p{N}]+| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|        |    |   |  |\s+(?!\S)|\s+"""

def get_word_counts(texts):
    SPLIT_PATTERN=re.compile(O200_K_PATTERN)
    word_counts = Counter()
    
    for text in tqdm(texts):
        word_counts.update(re.findall(SPLIT_PATTERN, text))
    
    del texts
    return word_counts


class WordNode():
    def __init__(self,token=None,next=None,prev=None,count=None):
        self.token=token
        self.next=next
        self.prev=prev
        self.count=count    
        
class WordList():
    def __init__(self,text=None,count=None,index_map=None):
        text=text.encode('utf-8')
        byte=list(bytes(text))
        self.head=None
        self.count=count
        self.head=WordNode((byte[0]),count=self.count)
        curr=self.head
        
        if len(byte)>1:
            i=0
            index_map[(byte[i],byte[i+1])]["pos"].append(self.head)
            index_map[(byte[i],byte[i+1])]["count"]+=count

            for i in range(1,len(byte)-1):
                node=WordNode((byte[i]),count=self.count)
                curr.next=node
                node.prev=curr
                curr=node
                
                index_map[(byte[i],byte[i+1])]["pos"].append(node)
                index_map[(byte[i],byte[i+1])]["count"]+=count
            
            node=WordNode((byte[len(byte)-1]),count=self.count)
            curr.next=node
            node.prev=curr
            curr=node
                            

class Index_Heap():
    def __init__(self,word_counts):
        self.heap=[]
        self.index_map=defaultdict(lambda: {"pos": [], "count": 0,"time":0})        
        self.wordchain=[]    
        self.counter=0
        self.time_hash={}
        self.modified=set()
        

        for text in word_counts:
            self.wordchain.append(WordList(text,word_counts[text],self.index_map))

        for token_pair in self.index_map:
            count=-self.index_map[token_pair]['count']
            self.heap.append((count,0,token_pair))

        heapq.heapify(self.heap)

    
    def heap_add(self,token_pair,count):
        self.counter+=1
        heapq.heappush(self.heap,(-count,-self.counter,token_pair))
        self.time_hash[token_pair]=-self.counter
        

    def merge(self,node,token_pair):
            token1,token2=token_pair
            if node.token==token1 and node.next.token==token2:
                node.token=(token1,token2)
                right=node.next

                if node.prev:
                    self.index_map[(node.prev.token,token1)]["count"]-=node.prev.count                    
                    self.index_map[(node.prev.token,node.token)]["pos"].append(node.prev)
                    self.index_map[(node.prev.token,node.token)]["count"]+=node.prev.count
                    self.modified.add((node.prev.token,node.token))
                    self.modified.add((node.prev.token,token1))
                
                if node.next.next:
                    self.index_map[(token2,node.next.next.token)]["count"]-=node.next.next.count                    
                    node.next.next.prev=node

                    self.index_map[(node.token,node.next.next.token)]["pos"].append(node)
                    self.index_map[(node.token,node.next.next.token)]["count"]+=node.next.next.count
                    self.modified.add((node.token,node.next.next.token))
                    self.modified.add((token2,node.next.next.token))
                
                node.next=node.next.next
                right.next=None
                right.prev=None
                return
            

    def get_max(self):
        while self.heap:
            
            count,time,token_pair=heapq.heappop(self.heap)          
            if count<0 and time==self.time_hash.get(token_pair,0):
                    return token_pair
        return None
    
    
    def update_heap(self):
        for token_pair in self.modified:
            self.heap_add(token_pair,self.index_map[token_pair]["count"])
    
    
    def update_index(self,token_pair):
        self.modified=set()

        for node in self.index_map[token_pair]["pos"]:
            if node.next and node.token==token_pair[0] and node.next.token==token_pair[1]:
                self.merge(node,token_pair)
        
        self.modified.discard(token_pair)
        self.update_heap()
        del self.index_map[token_pair]
        return


def arr_from_tuples_rec(tuple):
    if type(tuple)==int:
        return [tuple]
    else:
        tuple1,tuple2=tuple
        return arr_from_tuples_rec(tuple1) + arr_from_tuples_rec(tuple2)

arr=arr_from_tuples_rec( (((104, 101), 108), ((108, 111), 32)))
bytes(arr),bytes([125])


def build_vocab(texts,is_count=True,vocab_size=48000):
    if is_count:
        word_counts=texts
    else:
        word_counts=get_word_counts(texts)

    print("BUILDING INDEX HEAP")
    index_heap=Index_Heap(word_counts)   
    print("INDEX HEAP BUILT") 
    
    vocab={}
    del texts

    for i in range(256):
        vocab[bytes([i])]=i	
    
    for j in tqdm(range(i+1,vocab_size)):
        maxm=index_heap.get_max()

        if maxm:
            maxm_bytes=bytes(arr_from_tuples_rec(maxm))
            vocab[maxm_bytes]=j
        else:
            break

        index_heap.update_index(maxm)
    return vocab        
        

class Tokenizer():
    def __init__(self,vocab_size):
        self.vocab_size=vocab_size
        self.vocab=None
        self.encoding=None
        self.pat_str=O200_K_PATTERN

    def train(self,texts,is_count=False):
        self.vocab=build_vocab(texts,is_count,self.vocab_size)
        self.encoding = tiktoken.Encoding(
            name="custom_encoding",
            pat_str=self.pat_str,
            mergeable_ranks=self.vocab,
            special_tokens={
                "<|endoftext|>": len(self.vocab),
                "<|pad|>":       len(self.vocab)+1,
                "<|unk|>":       len(self.vocab)+2,
                "<think>":       len(self.vocab)+3,
                "</think>":      len(self.vocab)+4,
                "<|startofmath|>": len(self.vocab)+5,
                "<|endofmath|>":   len(self.vocab)+6,
                "<|box|>":         len(self.vocab)+7,
                "<|system|>":      len(self.vocab)+8,
                "<|user|>":        len(self.vocab)+9,
                "<|assistant|>":   len(self.vocab)+10,
            }
        )
    
    def encode(self,text):
        if self.encoding:
            tokens = self.encoding.encode(text,disallowed_special=())
            return tokens
        else:
            print("TRAIN THE TOKENIZER FIRST ")
    
    def decode(self,tokens):
        if self.encoding:
            text = self.encoding.decode(tokens)
            return text
        else:
            print("TRAIN THE TOKENIZER FIRST ")        

    def save_tokenizer(self,path):
        with open(path, "w") as f:
            for tok, rank in sorted(self.vocab.items(), key=lambda kv: kv[1]):  
                f.write(f"{base64.b64encode(tok).decode()} {rank}\n")
    
    def load_tokenizer(self,path):
        mergeable_ranks = {}
        with open(path) as f:
            for line in f:
                b64, rank = line.split()
                mergeable_ranks[base64.b64decode(b64)] = int(rank)

        self.vocab=mergeable_ranks
        self.encoding = tiktoken.Encoding(
            name="custom_encoding",
            pat_str=self.pat_str,
            mergeable_ranks=self.vocab,
            special_tokens={
                "<|endoftext|>": len(self.vocab),
                "<|pad|>":       len(self.vocab)+1,
                "<|unk|>":       len(self.vocab)+2,
                "<think>":       len(self.vocab)+3,
                "</think>":      len(self.vocab)+4,
                "<|startofmath|>": len(self.vocab)+5,
                "<|endofmath|>":   len(self.vocab)+6,
                "<|box|>":         len(self.vocab)+7,
                "<|system|>":      len(self.vocab)+8,
                "<|user|>":        len(self.vocab)+9,
                "<|assistant|>":   len(self.vocab)+10,
            }
        )
        return
    
if __name__=="__main__":

    tokenizer=Tokenizer(48000)
    word_counts = Counter(pickle.load(open("/home/rohit1/LLM_train/raw_corpus_optimized/count.pkl", "rb"))) 
    tokenizer.train(word_counts,True)
    tokenizer.save_tokenizer("tokenizer_vocab.titoken")
    
    text = "BIRTH OF THE TOKENIZATION GOD 123456 123"
    tokens = tokenizer.encode(text)
    print("Token IDs:", tokens)
    
    # print(tokenizer.vocab)
    decoded_text = tokenizer.decode(tokens)
    print("Decoded Text:", decoded_text)
