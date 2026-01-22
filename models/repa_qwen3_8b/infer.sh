#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpu
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --job-name=RePa-Qwen3-8B-infer
#SBATCH --account=a_eecs_ds
#SBATCH --time=02:00:00
#SBATCH --array=1-9
#SBATCH -o RePa_Qwen3-8B_infer_r0.%a.out
#SBATCH -e RePa_Qwen3-8B_infer_r0.%a.err

module load miniconda3
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate llama_factory_rocm

cd /scratch/user/uqxxu16/LLaMA-Factory

export HF_TOKEN=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
export WORK_DIR=/scratch/user/uqxxu16/LLaMA-Factory
export HF_HOME=$TMPDIR/hf_cache/${SLURM_ARRAY_TASK_ID}
export HF_DATASETS_CACHE=$TMPDIR/hf_datasets/${SLURM_ARRAY_TASK_ID}
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=0ade9853dc308fb5e9cccff325a70f046904c2cb
export WANDB_PROJECT="RePa-Qwen3-8B-C4-FT"
export WANDB_ENTITY="ackesnal-ai"

python ${WORK_DIR}/models/repa_qwen3_8b/stage1_infer_and_save.py \
    --model_dir ${WORK_DIR}/models/repa_qwen3_8b/original \
    --out_dir ${WORK_DIR}/models/repa_qwen3_8b/inferred_ratio0.${SLURM_ARRAY_TASK_ID} \
    --seq_len 2048 \
    --max_steps 32768 \
    --linear_ratio 0.${SLURM_ARRAY_TASK_ID}

cp ${WORK_DIR}/models/repa_qwen3_8b/modeling_qwen3.py ${WORK_DIR}/models/repa_qwen3_8b/inferred_ratio0.${SLURM_ARRAY_TASK_ID}/modeling_qwen3.py