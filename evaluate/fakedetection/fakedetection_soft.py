#!/usr/bin/env python3
import json
import sys
import math

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def map_label(label_str):
    if label_str == "real":
        return "bonafide", 1
    elif label_str == "fake":
        return "spoof", 0
    else:
        return "unknown", -1


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Error] Line {line_num}: JSON decode error {e}", file=sys.stderr)
                continue
            try:

                audio_path = data["audios"][0]
                rel_audio = "/".join(audio_path.split("/")[-3:])  # audio/zh/...wav


                label_str = data.get("labels", "unknown")
                mapped_label, true_label_bin = map_label(label_str)

                content_list = (data.get("logprobs") or {}).get("content") or []
                if content_list:

                    for item in reversed(content_list):
                        if item.get("token") in ("real", "fake"):
                            top_logprobs = item.get("top_logprobs", [])

                # top_logprobs
                token2logprob = {item["token"]: item["logprob"] for item in top_logprobs}


                logit_real = token2logprob.get("real", -100)
                logit_fake = token2logprob.get("fake", -100)
                probs = softmax([logit_real, logit_fake])
                prob_real = probs[0]


                pred_bin = 1 if prob_real >= 0.5 else 0
                pred_label = "bonafide" if pred_bin == 1 else "spoof"
                true_label = "bonafide" if true_label_bin == 1 else "spoof"


                fout.write(
                    f"unknown {rel_audio} {prob_real:.16f} {pred_bin} {true_label_bin} {true_label}\n"
                )

            except Exception as e:
                print(f"[Error] Line {line_num}: {e}. Data = {line}", file=sys.stderr)
                continue

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.jsonl output.txt", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    process_file(input_path, output_path)
