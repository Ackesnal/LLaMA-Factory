import os
import json
import time
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16")  # bfloat16/float16/float32
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    
    # Adjust mask according to the channel summarization
    model.reparam()
    model.config.reparamed = True
    
    # Save model & tokenizer (even though weights unchanged)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)

    print("[stage3] done. saved to:", args.out_dir)

if __name__ == "__main__":
    main()