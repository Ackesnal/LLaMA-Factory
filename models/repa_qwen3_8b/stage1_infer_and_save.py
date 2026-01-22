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

    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_steps", type=int, default=20000)   # 控制样本量（越大越“足够大”）
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_buffer", type=int, default=10000)
    ap.add_argument("--linear_ratio", type=float, default=0.1)

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
    model.set_attn_implementation("sdpa")
    model.eval()

    # Streaming C4-en
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    it = iter(ds)

    total_tokens = 0
    t0 = time.time()
    
    # Set scaler to all 0s
    model.init_scaler()
    model.adjust_linear_ratio(args.linear_ratio)
    model.config.linear_ratio = args.linear_ratio
    
    with torch.no_grad():
        for step in range(1, args.max_steps + 1):
            ex = next(it)
            text = ex.get("text", "")
            if not text:
                continue

            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.seq_len
            )
            input_ids = enc["input_ids"].to(args.device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(args.device)

            # 纯 forward（可选：计算 loss）
            _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

            total_tokens += int(input_ids.numel())

            if step % 200 == 0:
                dt = time.time() - t0
                print(f"[stage1] step={step:{len(str(args.max_steps))}d}/{args.max_steps}  tokens={total_tokens:,}  elapsed={int(dt/60):2d} min {int(dt%60):2d} sec")
    
    # Adjust mask according to the channel summarization
    model.adjust_mask()
    model.config.summarize_act = False
    
    # Save model & tokenizer (even though weights unchanged)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)

    meta = {
        "source_model_dir": args.model_dir,
        "seq_len": args.seq_len,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "dtype": args.dtype,
        "total_tokens": total_tokens,
    }
    with open(os.path.join(args.out_dir, "stage1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[stage1] done. saved to:", args.out_dir)

if __name__ == "__main__":
    main()