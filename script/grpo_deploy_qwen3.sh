# usage example:
# CUDA_VISIBLE_DEVICES=0 \
# bash grpo_deploy_qwen3.sh \
# "pretrain_model/Qwen3-8B"

MODEL_PATH=$1
swift deploy \
    --model "$MODEL_PATH" \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --served_model_name "Qwen3-8B" \
    --response_prefix "</think>\n\n" \
    --device_map auto \
