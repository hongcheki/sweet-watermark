import os
import json
from copy import deepcopy

import pandas as pd
import sklearn.metrics as metrics
import numpy as np

from transformers import HfArgumentParser
from lm_eval.tasks import ALL_TASKS

from lm_eval.arguments import EvalArguments

def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--task",
        choices=ALL_TASKS,
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--human_fname",
        type=str,
        default="outputs_human",
        help="File name of human code detection results",
    )
    parser.add_argument(
        "--machine_fname",
        type=str,
        default="outputs",
        help="File name of machine code detection results",
    )
    return parser.parse_args()

def get_roc_aur(human_z, machine_z):
    assert len(human_z) == len(machine_z)

    baseline_z_scores = np.array(human_z)
    watermark_z_scores = np.array(machine_z)
    all_scores = np.concatenate([baseline_z_scores, watermark_z_scores])

    baseline_labels = np.zeros_like(baseline_z_scores)
    watermarked_labels = np.ones_like(watermark_z_scores)
    all_labels = np.concatenate([baseline_labels, watermarked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    #print(tpr)
    #print(fpr)
    #print(thresholds)
    
    return roc_auc, fpr, tpr, thresholds

def get_tpr(fpr, tpr, error_rate):
    assert len(fpr) == len(tpr)

    value = None
    for f, t in zip(fpr, tpr):
        if f <= error_rate:
            value = t
        else:
            assert value is not None
            return value
        
    assert value == 1.0
    return value

def main():
    args = parse_args()

    human_results = json.load(open(args.human_fname))
    machine_results = json.load(open(args.machine_fname))

    # AUROC
    human_z = [r['z_score'] for r in human_results[args.task]['watermark_detection']['raw_detection_results']]
    machine_z = [r['z_score'] for r in machine_results[args.task]['watermark_detection']['raw_detection_results']]
    roc_auc, fpr, tpr, _ = get_roc_aur(human_z, machine_z)
    print(roc_auc)

    # TPR (FPR = 0%)
    tpr_value0 = get_tpr(fpr, tpr, 0.0)
    print(tpr_value0)

    # TPR (FPR = 1%)
    tpr_value1 = get_tpr(fpr, tpr, 0.01)
    print(tpr_value1)

    # TPR (FPR = 5%)
    tpr_value5 = get_tpr(fpr, tpr, 0.05)
    print(tpr_value5)

    # update metrics
    print(machine_results[args.task].keys())
    watermark_detection = deepcopy(machine_results[args.task]['watermark_detection'])
    raw_detection_results = watermark_detection.pop('raw_detection_results')
    watermark_detection['roc_auc'] = roc_auc
    watermark_detection['TPR (FPR = 0%)'] = tpr_value0
    watermark_detection['TPR (FPR < 1%)'] = tpr_value1
    watermark_detection['TPR (FPR < 5%)'] = tpr_value5
    watermark_detection['raw_detection_results'] = raw_detection_results
    watermark_detection['neg_samples_file'] = args.human_fname
    machine_results[args.task]['watermark_detection'] = watermark_detection

    json.dump(machine_results, open(args.machine_fname, 'w'), indent=4)

if __name__ == "__main__":
    main()