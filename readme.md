# SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation
[![arXiv](https://img.shields.io/badge/arXiv-2510.14664-b31b1b.svg)](https://arxiv.org/pdf/2510.14664) [![Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d.svg?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/Hui519/SpeechEval)

SpeechLLM-as-Judges provides an LLM-based pipeline for assessing speech quality with interpretable reasoning traces.

## 1. Environment Setup
We provide a `requirements.txt` to simplify dependency installation.

```bash
conda create -n speecheval python=3.10
conda activate speecheval
pip install -r requirements.txt
```

For Qwen3-8B deployment, we provide a `requirements_qwen3.txt`:

```bash
conda create -n qwen3_reward python=3.10
conda activate qwen3_reward
pip install -r requirements_qwen3.txt
```

## 2. Pretrained Checkpoints & Inference

Download the latest checkpoints from [Google Drive](https://drive.google.com/file/d/1buq7L1nHKHYZNooFQeXeU2Un1RkepKI2/view?usp=drive_link) and unzip them into your workspace. Suppose the adapter weights reside at `path/to/checkpoint`.

Run inference with swift:

```bash
cd script/
CUDA_VISIBLE_DEVICES=0 \
bash inference.sh \
"path/to/checkpoint" \
"swift_style_test/split_data_single_eval/test/multitask.jsonl" \
"path/to/checkpoint/results_inference.jsonl"
```

Run inference (`fake detection`) with swift:

```bash
cd script/
CUDA_VISIBLE_DEVICES=0 \
bash inference.sh \
"path/to/checkpoint" \
"swift_style_test/split_data_single_eval/test/multitask.jsonl" \
"path/to/checkpoint/results_inference.jsonl"
```

Change `CUDA_VISIBLE_DEVICES`, the adapter path, or the evaluation JSON as needed. The generated results are stored in `path/to/checkpoint/results_inference.jsonl`.

## 3. Training
The training data is hosted on [Hugging Face](https://huggingface.co/datasets/Hui519/SpeechEval).

### SFT
```bash
cd script/
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash qwenomni_train_ratio.sh \
"pretrain_model/Qwen25-Omni-7B" \
"checkpoint/" \
"data_dev" \
setting_ratio_thinkcot \
'["SingleEval","CompareEval","Suggest","FakeDetection"]' \
v0 \
8 \
50 \
23450 \
4 \
'[1.0,1.0,1.0,3.0]'
```

### GRPO
Deploy Qwen3-8B for GRPO training:
```bash
cd script/
CUDA_VISIBLE_DEVICES=0 \
bash grpo_deploy_qwen3.sh \
"pretrain_model/Qwen3-8B"
```
GRPO training:
```bash
cd script/
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash grpo_train.sh \
"checkpoint-2000-merged" \
"all_grpo_think_format/train.jsonl" \
"all_grpo_think_format/val.jsonl#1000" \
"evaluate/plugin/plugin.py" \
"checkpoint/qwenomni_grpo_cot_4_reward" \
"external_my_multitask_qwen_help,external_my_multitask_qwen_relevance,external_my_multitask_qwen_accuracy,external_my_multitask_qwen_detail" \
1,1,2,0.5 \
4 \
```

## 4. Evaluation
For deepseek api evaluation, you can follow the instructions in `evaluate/`.

For fake detection evaluation, use:
```bash
cd evaluate/fake_detection/
python fakedetection_soft.py \
    output.jsonl \
    fake_detection_eval.txt
    
python calculate_metrics.py \
    fake_detection_eval.txt \
    fake_detection_metrics.txt

```

## Acknowledgements
This repository builds upon the `swift` toolkit. We thank the original authors and open-source community for their contributions.

## Reference
If this project helps your research, please consider citing:

```
@article{wang2025speechllm,
  title={SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation},
  author={Wang, Hui and Zhao, Jinghua and Yang, Yifan and Liu, Shujie and Chen, Junyang and Zhang, Yanzhe and Zhao, Shiwan and Li, Jinyu and Zhou, Jiaming and Sun, Haoqin and others},
  journal={arXiv preprint arXiv:2510.14664},
  year={2025}
}
```
