import os
import numpy as np

from calculate_modules import *


def calculate_minDCF_EER_ACC(cm_scores_file,
                       output_file,
                       printout=True):
    # Evaluation metrics for Phase 1
    # Primary metrics: min DCF,
    # Secondary metrics: EER, Accuracy (replacing CLLR)

    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }


    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    
    # 检查数据格式，确定列的位置
    if cm_data.shape[1] >= 6:  # 新格式: spk_id utt_id score pred true_label key
        cm_keys = cm_data[:, 5]  # 第6列是标签类型 (bonafide/spoof)
        cm_scores = cm_data[:, 2].astype(np.float64)  # 第3列是得分
        cm_preds = cm_data[:, 3].astype(np.int32)  # 第4列是预测标签
        cm_true_labels = cm_data[:, 4].astype(np.int32)  # 第5列是真实标签
    elif cm_data.shape[1] >= 4:  # 旧格式: spk_id utt_id score key
        cm_keys = cm_data[:, 3]  # 第4列是标签 (bonafide/spoof)
        cm_scores = cm_data[:, 2].astype(np.float64)  # 第3列是得分
        # 在旧格式中，我们没有预测标签和真实标签，因此无法计算准确率
        print("[calculate_metrics] Warning: Old format detected. Cannot calculate accuracy without prediction and true labels.")
        cm_preds = None
        cm_true_labels = None
    else:
        # 如果格式不同，尝试适应
        print(f"[calculate_metrics] Warning: Unexpected CM scores file format with {cm_data.shape[1]} columns")
        if cm_data.shape[1] == 3:  # 假设格式是 utt_id score key
            cm_keys = cm_data[:, 2]
            cm_scores = cm_data[:, 1].astype(np.float64)
        elif cm_data.shape[1] == 2:  # 假设格式是 score key
            cm_keys = cm_data[:, 1]
            cm_scores = cm_data[:, 0].astype(np.float64)
        else:
            raise ValueError(f"[calculate_metrics] Warning: Unsupported CM scores file format: {cm_data.shape[1]} columns")
        cm_preds = None
        cm_true_labels = None

    # 过滤掉标记为 'unknown' 的样本
    valid_indices = cm_keys != 'unknown'
    # 如果过滤了，输出过滤掉的个数
    if not valid_indices.all():
        print(f"[calculate_metrics] Warning: Filtered out {len(cm_keys) - valid_indices.sum()} samples")
    cm_keys = cm_keys[valid_indices]
    cm_scores = cm_scores[valid_indices]
    if cm_preds is not None:
        cm_preds = cm_preds[valid_indices]
    if cm_true_labels is not None:
        cm_true_labels = cm_true_labels[valid_indices]

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    
    # 确保有足够的样本进行计算
    if len(bona_cm) == 0 or len(spoof_cm) == 0:
        print("Warning: Not enough samples for calculation")
        print(f"bonafide samples: {len(bona_cm)}, spoof samples: {len(spoof_cm)}")
        if len(bona_cm) == 0 or len(spoof_cm) == 0:
            # 如果没有足够的样本，返回默认值
            return 1.0, 0.5, 0.0  # minDCF, EER, ACC

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)#[0]
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])
    
    # 计算准确率（替代CLLR）
    if cm_preds is not None and cm_true_labels is not None:
        # 只计算有效的样本（真实标签不为-1）
        valid_labels = cm_true_labels != -1
        if np.sum(valid_labels) > 0:
            accuracy = np.mean(cm_preds[valid_labels] == cm_true_labels[valid_labels])
        else:
            print("Warning: No valid labels found for accuracy calculation")
            accuracy = 0.0
    else:
        # 如果没有预测标签或真实标签，则无法计算准确率
        print("Warning: Cannot calculate accuracy without prediction and true labels")
        accuracy = 0.0

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tmin DCF \t\t= {} % '
                        '(min DCF for countermeasure)\n'.format(
                            minDCF_cm))
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\tACC\t\t= {:8.9f} % '
                        '(Accuracy for countermeasure)\n'.format(
                            accuracy * 100))
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm, accuracy

if __name__=="__main__":
    if len(sys.argv)!=3:
        print(f"Usage: {sys.argv[0]} <input_txt> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_txt = sys.argv[1]
    output_file = sys.argv[2]

    calculate_minDCF_EER_ACC(input_txt, output_file)