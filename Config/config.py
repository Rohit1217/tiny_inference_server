import os
from dataclasses import dataclass


@dataclass
class Config:
    BATCH_SIZE:int = 24
    SEQ_LEN:int = 1024
    WORLD_SIZE:int = int(os.environ.get("WORLD_SIZE", 1))   #torchrun sets WORLD_SIZE = nproc_per_node*nnodes; matches launch automatically

    D_MODEL:int = 1536
    VOCAB_SIZE:int = 48011 #48000 BPE TOKENS (0..47999) + SPECIAL TOKENS 
    MAX_CONTEXT:int = 8192
    MAX_FREQ:int = 10000
    N_HEAD:int = 12
    NUM_LAYERS:int = 30
    FFN_HIDDEN_DIM:int = 4096

    ATT_DROPOUT:float = 0
    FFN_DROPOUT:float = 0

    NUM_GROUPS:int=4
    MTP_HEADS:int=2
    MTP_LOSS_WEIGHT:float=0.3

    GRAD_CHECKPOINT_EVERY:int=0
    ACCUMULATION_STEP_REF:int=12   #micro-batches/opt-step per rank AT WORLD_REF ranks. Actual ACCUMULATION_STEP is derived
    WORLD_REF:int=4                #world size the token budget + LR schedule are calibrated for

    WEIGHT_DECAY:float=0.1
    LR:float=2e-4

    DEVICE:str = "cuda:0"
    SEED:int = 133721
    RUN_NAME:str = os.environ.get("RUN_NAME", "llm_mtp_intradoc_main")
    DATA_DIR:str = os.environ.get("LLM_DATA_DIR", "raw_corpus_optimized")
    OVERFIT_FILE_PATH:str = os.path.join(DATA_DIR, "overfit_10k_sequences.bin")
    MAIN_FILE_PATH:str = os.path.join(DATA_DIR, "tokens.bin")

    WANDB_OFFLINE:bool = False             
    RUN_ACT_STATS:bool = False            
    CKPT_DIR:str = "Weights/checkpoints"  
    MAX_CHECKPOINTS:int = 25              
    CKPT_EVERY_OPT:int = 400              
    RESUME_CHECKPOINT:str = ""            


    INTRADOC_ATT=True

    TOTAL_TOKENS:int = 2*(10**10) #20 billion tokens
    A6000_BF16_PEAK :float= 154e12
    THROUGHPUT_TARGET:float = 15805   # tok/s baseline
    PEAK_MFU_TARGET:float = 0.563

    def __post_init__(self):
        self.EFF_SEQ_LEN:int=self.SEQ_LEN+self.MTP_HEADS+1

        #AUTO-SCALE ACCUM inversely with world so GLOBAL batch (tokens/opt-step) is CONSTANT across world sizes.

        assert (self.ACCUMULATION_STEP_REF*self.WORLD_REF) % self.WORLD_SIZE == 0, \
            f"WORLD_SIZE={self.WORLD_SIZE} must divide ACCUMULATION_STEP_REF*WORLD_REF={self.ACCUMULATION_STEP_REF*self.WORLD_REF}"
        self.ACCUMULATION_STEP:int = self.ACCUMULATION_STEP_REF*self.WORLD_REF // self.WORLD_SIZE

        self.GLOBAL_BATCH_SEQS = self.BATCH_SIZE*self.WORLD_SIZE*self.ACCUMULATION_STEP   
        self.TOKENS_PER_STEP   = self.BATCH_SIZE*self.SEQ_LEN*self.WORLD_SIZE            

        self.OPT_STEP:int   = self.TOTAL_TOKENS // (self.GLOBAL_BATCH_SEQS*self.SEQ_LEN)  
        self.TOTAL_STEPS    = self.OPT_STEP * self.ACCUMULATION_STEP                       

        self.CKPT_DIR = os.path.join(self.CKPT_DIR, self.RUN_NAME)

        self.WARMUP_STEPS = int(0.05*self.OPT_STEP) 
        self.STABLE_STEPS = int(0.85*self.OPT_STEP)
        self.DECAY_STEPS  = int(0.10*self.OPT_STEP)