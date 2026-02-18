#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --account=12345678
#SBATCH --time=00-23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=accel
#SBATCH --gpus=2


module load Miniconda3/22.11.1-1
export PS1=\$
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
echo "Conda environments: $(conda info --envs)"
echo "EBROOTMINCONDA3: ${EBROOTMINICONDA3}"
module load CUDA/12.x.0
conda activate /path/to/your/conda/env

export WANDB_MODE=disabled

direction="i2c" # i (issue) refer to email discussion, and c (code) is the I-D/RFC; query is the issue, passage is the code in this case.
# or "c2i" if you want to treat code as query and issue as passage; i2c and c2i do not have the same training data.

# Update this path to match your downloaded dataset
train_data="../dataset/CodeConvo/train/$direction"

# set large epochs and small batch size for testing
num_train_epochs=5
per_device_train_batch_size=24

# set num_gpus to 2 for testing
num_gpus=2

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval '' \
    --query_instruction_format '{}query: {}' \
    --knowledge_distillation False \
"

training_args="\
    --output_dir ./output_encoder_models/ \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../../ds_stage0.json \
    --logging_steps 50 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --save_strategy 'epoch' \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd