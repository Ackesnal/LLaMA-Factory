#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpu
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:4
#SBATCH --cpus-per-task=80
#SBATCH --mem=512gb
#SBATCH --job-name=RePa-Mistral-7B-finetune
#SBATCH --account=a_eecs_ds
#SBATCH --time=1-00:00:00
#SBATCH --array=9-9
#SBATCH -o RePa_Mistral-7B_finetune_r0.%a.out
#SBATCH -e RePa_Mistral-7B_finetune_r0.%a.err

module load miniconda3
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate llama_factory_rocm

cd /scratch/user/uqxxu16/LLaMA-Factory


llamafactory-cli train ${WORK_DIR}/models/repa_mistral_7b/repa_mistral-7b_r0.${SLURM_ARRAY_TASK_ID}_full_sft.yaml

cp ${WORK_DIR}/models/repa_mistral_7b/modeling_mistral.py ${WORK_DIR}/models/repa_mistral_7b/finetuned_ratio0.${SLURM_ARRAY_TASK_ID}/modeling_mistral.py