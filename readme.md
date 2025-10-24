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

## 2. Pretrained Checkpoints & Inference
Download the latest checkpoints from [Google Drive](https://drive.google.com/file/d/1buq7L1nHKHYZNooFQeXeU2Un1RkepKI2/view?usp=drive_link) and unzip them into your workspace. Suppose the adapter weights reside at `path/to/checkpoint`.

Run inference with `swift`:

```bash
CUDA_VISIBLE_DEVICES=5 \
swift infer \
  --model_type qwen2_5_omni \
  --adapters path/to/checkpoint \
  --val_dataset example/test.json \
  --temperature 0 \
  --max_batch_size 2 \
  --max_new_tokens 2048 \
  --result_path example/output.json
```

Change `CUDA_VISIBLE_DEVICES`, the adapter path, or the evaluation JSON as needed. The generated results are stored in `example/output.json`.

## 3. Training
The training data is hosted on [Hugging Face](https://huggingface.co/datasets/Hui519/SpeechEval).

### Stage I
```bash
bash train_stage1.sh
```

### Stage II
```bash
bash train_stage2.sh
```

## 4. Evaluation
```bash
bash test.sh
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
