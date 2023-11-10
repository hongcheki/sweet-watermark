import datasets
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, LlamaForCausalLM

from lm_eval import tasks
from lm_eval.arguments import EvalArguments
from lm_eval.tasks import ALL_TASKS
from lm_eval.utils import calculate_entropy
from main import pattern_match

def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="bigcode/starcoder",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="for LLaMA, the local path of llama ckpts is needed",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS,
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    assert args.task is not None
    task_name = pattern_match([args.task], ALL_TASKS)[0]

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Task: {task_name}")

    # load tokenizer

    if args.model != "llama":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            truncation_side="left",
            padding_side="right",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.precision not in dict_precisions:
        raise ValueError(
            f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
        )

    # load model

    if args.model != "llama":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            torch_dtype=dict_precisions[args.precision],
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dict_precisions["fp16"],
        )
    model = model.to(accelerator.device)

    # load task

    task = tasks.get_task(task_name)
    dataset = task.get_dataset()
    n_tasks = args.limit if args.limit else len(dataset)

    # now, calcualte
    all_ents = []

    for data in tqdm(dataset, total=len(dataset[:n_tasks])):
        tokenized_input = tokenizer(
            task.get_full_data(data),
            padding=True,
            truncation=True,
            return_tensors="pt")['input_ids'].squeeze()
        
        tokenized_prompt = tokenizer(
            task.get_prompt(data),
            padding=True,
            truncation=True,
            return_tensors="pt")['input_ids'].squeeze()

        entropy = calculate_entropy(
            model,
            tokenized_input.to(accelerator.device))

        # we need to shift entropy to the right, so the first item is dummy
        entropy = [0] + entropy[:-1]

        all_ents += entropy[len(tokenized_prompt):]

    #plt.hist(all_ents, np.arange(0,3,0.1).tolist())
    #plt.savefig(f"{args.task}_{args.model.replace('/', '-')}_entropy_histogram.jpg")

    print(len(all_ents))
    print("mean :", np.mean(all_ents))
    print("1st quartile :", np.percentile(all_ents, 25))
    print("median :", np.percentile(all_ents, 50))
    print("3rd quartile :", np.percentile(all_ents, 75))

if __name__ == "__main__":
    main()