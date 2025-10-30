# usage example:
# CUDA_VISIBLE_DEVICES=1 \
# bash inference.sh \
# "v0-20250924-131627/checkpoint-2250" \
# "swift_style_test/split_data_single_eval/test/multitask.jsonl" \
# "v0-20250924-131627/checkpoint-2250/results_inference.jsonl"

# trained model adapter path
final_ckpt=$1
# val dataset
val_data_path=$2
# output jsonl path
output_jsonl=$3

VIDEO_MAX_PIXELS=50178 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
swift infer \
    --model_type qwen2_5_omni \
    --adapters $final_ckpt \
    --val_dataset $val_data_path \
    --temperature 0 \
    --max_new_tokens 512 \
    --logprobs True \
    --top_logprobs 10 \
    --write_batch_size 50 \
    --max_batch_size 8 \
    --result_path $output_jsonl