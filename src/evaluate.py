import os
from os.path import join, dirname
import json
import torch
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig,
)
from transformers.pipelines.pt_utils import KeyDataset
from human_eval.data import read_problems
from datasets import load_dataset
from dataclasses import dataclass, field
from tqdm import tqdm
from functools import partial


@dataclass
class EvalArguments:
    ckpt: str
    tokenizer: str
    batch_size: int
    eval_type: str = "humaneval"
    humaneval_x_dir: str = None  # humaneval_x only
    result_dir: str = "./result"
    output_name: str = "draft.jsonl"
    temprature: float = 0.8
    candidate_num: int = 5
    max_new_tokens: int = 700
    greedy_decode: bool = False
    inst_type: str = "v2"
    repetition_penalty: float = 1.0
    append: bool = False


class Tester:
    def __init__(self, args: EvalArguments) -> None:
        self.result_dir = args.result_dir
        self.output_name = args.output_name
        self.batch_size = args.batch_size
        self.temprature = args.temprature
        self.candidate_num = args.candidate_num
        self.max_new_tokens = args.max_new_tokens
        self.greedy_decode = args.greedy_decode
        self.inst_type = args.inst_type
        self.append = args.append
        self.repetition_penalty = args.repetition_penalty
        self.model = AutoModelForCausalLM.from_pretrained(
            args.ckpt,
            torch_dtype='auto',
            device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            # add_eos_token=True,
            pad_token_id=0,
            padding_side='left',
        )

    def humaneval(self):
        def generate_inst_(inst_type, prompt, **kwargs):
            desc = prompt.partition('"""')[2].partition('"""')[0].strip()
            inst_dict = {
                "v1": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Complete this Python function in a markdown code block to solve programming problem:
```python
{prompt}
```

### Response:""",
                "500": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Complete this Python function in a markdown code block to solve programming problem:
```python
{prompt}
```
Your response is limited to 500 tokens.

### Response:""",
                "cf": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
As an expert code developer with years of experience, please provide the source code based on the problem description. The detailed information are as follows:
1. Problem description: {desc}
2. Programming language: Python

### Response:\n{prompt}""",
                "newcf": f"""[INST]<<sys>>Below is an instruction that describes a task. Write a response that appropriately completes the request.
<</SYS>>As an expert code developer with years of experience, please provide the source code based on the problem description. The detailed information are as follows:
1. Problem description: {desc}
2. Programming language: Python
[/INST]{prompt}""",
                "no_prompt": {prompt},
            }
            return inst_dict[inst_type]
        generate_inst = partial(generate_inst_, inst_type=self.inst_type)

        gen_conf = GenerationConfig(
            pad_token_id=self.tokenizer.eos_token_id,
            # do_sample=(not self.greedy_decode),
            # top_p=0.95,
            temprature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.candidate_num,
            use_cache=True,
        )
        if self.greedy_decode:
            gen_conf.do_sample = False
        else:
            gen_conf.do_sample = True
            gen_conf.top_p = 0.95

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        pipe = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device_map='auto',
            return_full_text=False,
            batch_size=self.batch_size,
        )
        dataset = list(read_problems().values())
        for data in dataset:
            data.update({'inst': generate_inst(**data)})
        results = pipe(
            KeyDataset(dataset, 'inst'),
            generation_config=gen_conf,
        )
        out_file = join(self.result_dir, self.output_name)
        os.makedirs(dirname(out_file), exist_ok=True)
        mode = 'a' if self.append else 'w'
        with open(out_file, mode, encoding='utf-8') as fp:
            results_ = tqdm(results, desc='generating', total=len(dataset))
            for idx, res in enumerate(results_):
                task_id = dataset[idx]['task_id']
                # 一个res对应一个task，有多个生成样本
                for sample in res:
                    sample_dict = {
                        'task_id': task_id,
                        'completion': sample['generated_text']
                    }
                    fp.write(json.dumps(sample_dict, ensure_ascii=False)+'\n')

        print(f"write to {out_file}")


if __name__ == "__main__":
    parser = HfArgumentParser(EvalArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()

    tester = Tester(eval_args)
    if eval_args.eval_type == 'humaneval':
        tester.humaneval()
    if eval_args.eval_type == 'humaneval-x':
        tester.humaneval_x(eval_args.humaneval_x_dir)
