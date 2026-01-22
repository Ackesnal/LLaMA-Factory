#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=sxm
#SBATCH --partition=gpu_sxm
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=960gb
#SBATCH --job-name=RePa-LLaMA3-8B-finetune
#SBATCH --account=a_eecs_ds
#SBATCH --time=1-00:00:00
#SBATCH --array=8-8
#SBATCH -o RePa_LLaMA3-8B_finetune_r0.%a.out
#SBATCH -e RePa_LLaMA3-8B_finetune_r0.%a.err

module load miniconda3
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate llama_factory_cuda

cd /scratch/user/uqxxu16/LLaMA-Factory

export HF_TOKEN=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
export WORK_DIR=/scratch/user/uqxxu16/LLaMA-Factory
export HF_HOME=$TMPDIR/hf_cache
export HF_DATASETS_CACHE=$TMPDIR/hf_datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=0ade9853dc308fb5e9cccff325a70f046904c2cb
export WANDB_PROJECT="RePa-Llama3-8B-C4-FT"
export WANDB_ENTITY="ackesnal-ai"

llamafactory-cli train ${WORK_DIR}/models/repa_llama3_8b/repa_llama3-8b_r0.${SLURM_ARRAY_TASK_ID}_full_sft.yaml

cp ${WORK_DIR}/models/repa_llama3_8b/modeling_llama.py ${WORK_DIR}/models/repa_llama3_8b/finetuned_ratio0.${SLURM_ARRAY_TASK_ID}/modeling_llama.py