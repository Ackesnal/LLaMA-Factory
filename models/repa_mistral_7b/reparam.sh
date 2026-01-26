#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpu
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --job-name=RePa-Mistral-7B-reparam
#SBATCH --account=a_eecs_ds
#SBATCH --time=02:00:00
#SBATCH --array=1-8
#SBATCH -o RePa_Mistral-7B_reparam_r0.%a.out
#SBATCH -e RePa_Mistral-7B_reparam_r0.%a.err

module load miniconda3
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate llama_factory_rocm

cd /scratch/user/uqxxu16/LLaMA-Factory



python ${WORK_DIR}/models/repa_mistral_7b/stage3_reparam.py --model_dir /QRISdata/Q6719/RePaMistral/RePaMistral-7B/finetuned_ratio0.${SLURM_ARRAY_TASK_ID}/ --out_dir /QRISdata/Q6719/RePaMistral/RePaMistral-7B/reparamed_ratio0.${SLURM_ARRAY_TASK_ID}/


cp ${WORK_DIR}/models/repa_mistral_7b/modeling_mistral.py /QRISdata/Q6719/RePaMistral/RePaMistral-7B/reparamed_ratio0.${SLURM_ARRAY_TASK_ID}/modeling_mistral.py