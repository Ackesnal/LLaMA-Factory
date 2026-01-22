import argparse
import gc
import logging
import math
import time
import random
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from eval_evaluations import *
from eval_datasets import *


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

  logging.info(f"Seed for reproducibility: {seed}")


"""
Prints the number of parameters of the model
"""
def printModelStats(model, model_type):
  model_total_params = sum(p.numel() for p in model.parameters())
  main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
  logging.info(f"[{model_type}] Full number of parameters = {model_total_params}")
  logging.info(f"[{model_type}] Main model number of parameters = {main_model_total_params}")


def loadModel(model_name, cache_dir=None):
  logging.info("Loading the model")

  if (
    "llama" in model_name.lower()
    or "phi-3" in model_name.lower()
    or "qwen2" in model_name.lower()
  ):
    dtype = torch.bfloat16
  else:
    dtype = torch.float16

  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
    dtype=dtype,
    device_map="auto",
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
  )

  return model
  

def get_calibration(dataset, tokenizer, num_samples, seq_len=2048, seed=0):
  samples_indices = list(range(len(dataset)))
  if seed != 0:
    random.shuffle(samples_indices)

  samples = dataset.select(samples_indices)
  input_ids = tokenizer(
    "\n\n".join(samples["text"]), return_tensors="pt", add_special_tokens=False
  ).input_ids

  # Split the calibration into num_samples of seq_len length
  calibration = []
  for i in range(num_samples):
    calibration.append(input_ids[:, i * seq_len : (i + 1) * seq_len])

  return calibration
  
  
def parse_args():
  parser = argparse.ArgumentParser(description="Pruning of transformer models")
  parser.add_argument('--model', type=str, required=True, help="Specify the model's name or path to be pruned")
  parser.add_argument('--seed', type=int, default=0, help="Set a seed for reproducibility (default: 0)")
  parser.add_argument('--cache_dir', type=str, required=False, help="Path to a directory in which a downloaded pretrained model should be cached. This option is not supported when --pruning_method=slicegpt")

  parser.add_argument('--main_table_results', help="Generate results for the main results table in the paper (Table 1)", action='store_true')
  parser.add_argument('--evaluate_inference', help="Measure the model's inference time", action='store_true')
  parser.add_argument('--evaluate_downstream', help="Perform downstream task evaluation at 37.5%% sparsity", action='store_true')
  parser.add_argument('--evaluate_perplexity', help="Evaluates perplexity on Wikitext2 only", action='store_true')
  parser.add_argument('--evaluate_qualitative', help="Qualitative results", action='store_true')

  parser.add_argument('--local_datasets', help="Use local datasets stored in the './data/' folder", action='store_true')
  
  parser.add_argument(
        '--logging',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the logging level (default: INFO)"
    )

  return parser.parse_args()


@torch.no_grad()
def main():
  args = parse_args()
  logging_level = getattr(logging, args.logging.upper())
  logging.basicConfig(
      level=logging_level,
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%H:%M:%S'
  )
  
  set_seed(args.seed)
  
  # Load the tokenizer
  logging.info("Loading the tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
  logging.info("Loaded the tokenizer")

  ###################### Datasets 
  logging.info("Loading the Datasets")
    
  # Evaluation datasets
  dataset_wikitext = load_wikitext2(args.local_datasets)
  dataset_c4_val = load_c4(train=False, local=args.local_datasets)
  dataset_fineweb_edu = load_fineweb_edu(local=args.local_datasets)[:500]
  dataset_c4_train = load_c4(train=True, local=args.local_datasets)
  logging.info("Loaded the Datasets")

  logging.info("Tokenizing the Datasets")
  wikitext_input_ids = tokenizer("\n\n".join(dataset_wikitext["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  c4_val_input_ids = tokenizer("\n\n".join(dataset_c4_val["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  fineweb_edu_input_ids = tokenizer("\n\n".join(dataset_fineweb_edu["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  logging.info("Tokenized the datasets")
  
  

  logging.info("Dense model evaluation")
  logging.info("Loading the model")    
  model = loadModel(args.model, args.cache_dir)
  #model.set_attn_implementation("flash_attention_2")
  #model.init_scaler()
  logging.debug(model)
  printModelStats(model, "Dense model")
  
  if args.evaluate_inference == True:
    calibration_dataset = get_calibration(dataset_c4_train, tokenizer, num_samples=1, seq_len=2048)
    first_calibration_sample = calibration_dataset[0]
    
    evaluate_inference_time(model, first_calibration_sample)

  if args.evaluate_downstream == True:
    evaluation_downstream(model, args.model)

  if args.main_table_results == True:
    evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids)
  
  if args.evaluate_perplexity == True:
    ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
    logging.info(f"Perplexity (wikitext2): {ppl}")  

  if args.evaluate_qualitative == True:
    qualitative_results(model, tokenizer, max_length=128)
    

if __name__ == "__main__":
  main()