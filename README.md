# sweet-watermark

## Introduction
Official repository of the paper:

"[Who Wrote this Code? Watermarking for Code Generation](https://arxiv.org/abs/2305.15060)" by Taehyun Lee*, Seokhee Hong*, Jaewoo Ahn, Ilgee Hong, Hwaran Lee, Sangdoo Yun, Jamin Shin', Gunhee Kim'

<p align="center">
    <img src="./img/main_table.png" alt="main table" width="80%" height="80%"> 
</p>
<p align="center">
    <img src="./img/pareto_figure.png" alt="Pareto Frontier" width="80%" height="80%"> 
</p>

## Reproducing the Main Experiments

### 1. Calculating the entropy of the model for human-written code
Please refer to our paper for details.

```
accelerate launch calculate_human_entropy.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task {humaneval,mbpp} \
    --precision bf16
```

### 2. Generating watermarked machine-generated code and Detecting watermarks
**Important**: For rigorous reproduction, same `batch_size` should be used with ours. Nevertheless, we observed similar results in which SWEET outperforms baselines when different `batch_size` is used.

As described in our paper, we generated `n_samples=40` and `20` samples for HumanEval and MBPP, respectively. `batch_size` was used to the same value as the `n_samples`.

Note that we used different `hash_key` for MBPP which is `15485917`, not `15485863`(default). This was for debugging and we observed similar results when we used default hash key value. For MBPP, add `--hash_key 15485917` argument.

```
accelerate launch main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task {humaneval,mbpp} \
    --batch_size {40,20} \
    --precision bf16 \
    --allow_code_execution \
    --outputs_dir OUTPUT_DIRECTORY \
    --metric_output_path EVALUATION_RESULTS_FNAME \
    --save_generations \
    --save_generations_path GENERATION_RESULTS_FNAME \
    --wllm or --sweet \
    --gamma GAMMA \
    --delta DELTA \
    --entropy_threshold ENTROPY_THRESHOLD \
    --n_samples {40,20}
```

### 3. Detecting watermarks in human-written code
```
accelerate launch main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task {humaneval,mbpp} \
    --precision bf16 \
    --allow_code_execution \
    --outputs_dir OUTPUT_DIRECTORY \
    --metric_output_path EVALUATION_RESULTS_FNAME \
    --wllm or --sweet \
    --gamma GAMMA \
    --delta DELTA \
    --entropy_threshold ENTROPY_THRESHOLD \
    --detect_human_code
```

### 4. Calculating Metrics (AUROC, TPR)
With both metric output files from machine-generated and human-written codes, we finally calculate metrics including AUROC and TPR and update attach the results to `EVALUATION_RESULTS_FNAME_MACHINE`.

```
python calculate_auroc_tpr.py \
    --task {humaneval,mbpp} \
    --human_fname EVALUATION_RESULTS_FNAME_HUMAN \
    --machine_fname EVALUATION_RESULTS_FNAME_MACHINE
```

## Acknowledgements
This repository is based on the codes of [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) in [BigCode Project](https://github.com/bigcode-project).

## Contact
If you have any questions about our codes, feel free to ask us: Taehyun Lee (taehyun.lee@vision.snu.ac.kr) or Seokhee Hong (seokhee.hong@vision.snu.ac.kr)
