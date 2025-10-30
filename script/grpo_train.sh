# usage example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# bash grpo_train.sh \
# "checkpoint-2000-merged" \
# "all_grpo_think_format/train.jsonl" \
# "all_grpo_think_format/val.jsonl#1000" \
# "evaluate/plugin/plugin.py" \
# "checkpoint/qwenomni_grpo_cot_4_reward" \
# "external_my_multitask_qwen_help,external_my_multitask_qwen_relevance,external_my_multitask_qwen_accuracy,external_my_multitask_qwen_detail" \
# 1,1,2,0.5 \
# 4 \

# trained model path
MODEL=$1
TRAIN_DATA=$2
VAL_DATA=$3
# reward plugin path
EXTERNAL_PLUGINS=$4
OUTPUT_ROOT=$5
reward_list=$6
reward_list=${reward_list//,/ }
read -r -a REWARD_FUNCS <<< "$reward_list"
# reward weights
WEIGHT_RAW=$7
NODE=$8
WEIGHT_RAW=${WEIGHT_RAW//_/\.}
WEIGHT_RAW=${WEIGHT_RAW//,/ }
read -r -a REWARD_WEIGHTS <<< "$WEIGHT_RAW"

NPROC_PER_NODE=NODE \
VIDEO_MAX_PIXELS=50178 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
MASTER_PORT=$MASTER_PORT \
    swift rlhf \
      --rlhf_type grpo \
      --model "$MODEL" \
      --reward_funcs "${REWARD_FUNCS[@]}" \
      --reward_weights "${REWARD_WEIGHTS[@]}" \
      --train_type lora \
      --lora_rank 8 \
      --lora_alpha 32 \
      --target_modules all-linear \
      --torch_dtype bfloat16 \
      --dataset "$TRAIN_DATA" \
      --val_dataset "$VAL_DATA" \
      --system "You are a helpful assistant." \
      --external_plugins "$EXTERNAL_PLUGINS" \
      --max_completion_length 2048 \
      --max_steps 8000 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 8 \
      --learning_rate 1e-6 \
      --gradient_accumulation_steps $(16 / NPROC_PER_NODE) \
      --eval_steps 100 \
      --save_steps 100 \
      --save_total_limit 50 \
      --logging_steps 5 \
      --max_length 2048 \
      --output_dir "$OUTPUT_ROOT" \
      --warmup_ratio 0.05 \
      --dataloader_num_workers 8 \
      --dataset_num_proc 4 \
      --num_generations 4 \
      --temperature 1.0 \
      --top_p 0.99 \
      --top_k 50 \
      --log_completions true \
      --deepspeed zero2