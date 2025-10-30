# usage example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# bash qwenomni_train_ratio.sh \
# "pretrain_model/Qwen25-Omni-7B" \
# "checkpoint/" \
# "data_dev" \
# setting_ratio_thinkcot \
# '["SingleEval","CompareEval","Suggest","FakeDetection"]' \
# v0 \
# 8 \
# 50 \
# 23450 \
# 4 \
# '[1.0,1.0,1.0,3.0]'


PRETRAIN_MODEL=$1
DATA_ROOT=$2
CKPT_ROOT=$3

setting=$4         
datasets_str=$5    
version=$6         
epoch=$7               
step=$8              
master_port=$9   
nproc_per_node=${10:-4}
# datasets weights      
weights_str=${11:-null}



# 解析 dataset/weights
datasets_json=$(echo "$datasets_str" | jq -c '.')
if [ "$weights_str" = "null" ] || [ -z "${weights_str}" ]; then
    weights_json=$(echo "$datasets_json" | jq -c 'map(1)')
else
    weights_json=$(echo "$weights_str" | jq -c '.')
fi

readarray -t ds < <(echo "$datasets_json" | jq -r '.[]')
readarray -t ws < <(echo "$weights_json" | jq -r '.[]')

train_datasets=""
val_datasets=""

for i in "${!ds[@]}"; do
    name="${ds[$i]}"
    w="${ws[$i]:-1}"

    if [[ "$name" == cot* ]]; then
        subname=$(echo "$name" | cut -d'-' -f2)
        root="$DATA_ROOT/$(echo "$name" | cut -d'-' -f1)"
        train="$root/$subname/train.jsonl"
        val="$root/$subname/val.jsonl"
    else
        root="$DATA_ROOT/$name"
        train="$root/train.jsonl"
        val="$root/val.jsonl"
    fi

    L=$(wc -l < "$train")
    Nf=$(awk -v l="$L" -v w="$w" 'BEGIN{printf("%.10f", l*w)}')
    if awk -v w="$w" 'BEGIN{exit !(w>=1)}'; then
        N=$(awk -v n="$Nf" 'BEGIN{nf=n+0; print (nf==int(nf))?int(nf):int(nf)+1}')
    else
        N=$(awk -v n="$Nf" 'BEGIN{nf=n+0; x=int(nf); if (x<1) x=1; print x}')
    fi

    train_datasets="$train_datasets ${train}#${N}"
    val_datasets="$val_datasets ${val}"
done

echo "Train datasets: $train_datasets"
echo "Val datasets: $val_datasets"

setting_name="qwenomni_${setting}"
stage_name="1_$(printf "%s " "${ds[@]}" | tr ' ' '_' | tr -s '_' '_' | sed 's/_$//')"
out_dir="${CKPT_ROOT}/${setting_name}/${stage_name}"
echo "Output dir: $out_dir"


CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE="${nproc_per_node}" \
MASTER_PORT="${master_port}" \
VIDEO_MAX_PIXELS=50178 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
swift sft \
    --model ${PRETRAIN_MODEL} \
    --dataset ${train_datasets} \
    --val_dataset ${val_datasets} \
    --system 'You are a helpful assistant.' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs ${epoch} \
    --early_stop_interval 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(16 / NPROC_PER_NODE) \
    --eval_steps ${step} \
    --save_steps ${step} \
    --save_total_limit 50 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir ${out_dir} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed zero2 \
    --seed 42 \
    --dataset_shuffle true