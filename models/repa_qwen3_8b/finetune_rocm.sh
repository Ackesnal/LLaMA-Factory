#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpu
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:4
#SBATCH --cpus-per-task=256
#SBATCH --mem=1024gb
#SBATCH --job-name=RePa-Qwen3-8B-finetune
#SBATCH --account=a_eecs_ds
#SBATCH --time=1-00:00:00
#SBATCH --array=1,3,5,7,9
#SBATCH -o RePa_Qwen3-8B_finetune_r0.%a.out
#SBATCH -e RePa_Qwen3-8B_finetune_r0.%a.err

module load miniconda3
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate llama_factory_rocm

cd /scratch/user/uqxxu16/LLaMA-Factory


llamafactory-cli train ${WORK_DIR}/models/repa_qwen3_8b/repa_llama3_8b_r0.${SLURM_ARRAY_TASK_ID}_ft.yaml

cp ${WORK_DIR}/models/repa_qwen3_8b/modeling_qwen3.py ${WORK_DIR}/models/repa_qwen3_8b/finetuned_ratio0.${SLURM_ARRAY_TASK_ID}/modeling_qwen3.py