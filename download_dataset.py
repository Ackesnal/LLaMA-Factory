import os
os.environ["HF_DATASETS_CACHE"] = "data/c4_datasets"
os.environ["HF_HUB_CACHE"] = "data/c4_datasets"

from datasets import load_dataset
en = load_dataset("allenai/c4", "en", cache_dir="data/c4_datasets/cache", num_proc=20)
