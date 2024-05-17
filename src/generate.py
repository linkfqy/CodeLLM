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
from prompt import humaneval_inst, codescope_inst

env_map = {
    'c++': 'GNU C++17',
    'c#': 'MS C#',
    'java': 'Java 11',
    'javascript': 'JavaScript',
    'c': 'GNU C11',
    'python': 'Python 3',
    'php': 'PHP',
    'ruby': 'Ruby',
    'kotlin': 'Kotlin',
    'rust': 'Rust',
    'go': 'Go',
    'd': 'dmd 2.105.0 win32',
    'delphi': 'Delphi7 win32',
    'perl': 'Perl v5.20.3'
}


@dataclass
class GenArguments:
    ckpt: str
    tokenizer: str
    batch_size: int
    task: str = "humaneval"
    datapath: str = ""
    result_dir: str = "./result"
    output_name: str = "draft.jsonl"
    inst_type: str = "v2"
    temprature: float = 0.0
    candidate_num: int = 5
    max_new_tokens: int = 700
    greedy_decode: bool = False
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    append: bool = False
    use_process: bool = False


class Generator:
    def __init__(self, args: GenArguments) -> None:
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            args.ckpt,
            torch_dtype='auto',
            device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            # add_eos_token=True,
            # pad_token_id=0,
            padding_side='left',
            model_max_length=int(1e30),
        )
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def count_token(self, s):
        tokens = self.tokenizer(s)['input_ids']
        return len(tokens)

    def gen(self, dataset, inst_fn, jsoned):
        args = self.args
        gen_conf = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            # do_sample=(not self.greedy_decode),
            # top_p=0.95,
            temprature=args.temprature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            # num_return_sequences=args.candidate_num,
            use_cache=True,
        )
        if args.greedy_decode:
            gen_conf.do_sample = False
        else:
            gen_conf.do_sample = True
            gen_conf.top_k = args.top_k
            gen_conf.top_p = args.top_p
            gen_conf.num_return_sequences = args.candidate_num

        pipe = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device_map='auto',
            return_full_text=False,
            batch_size=args.batch_size,
        )
        for data in dataset:
            data.update({'inst': inst_fn(**data)})
            if (cnt := self.count_token(data['inst'])) > 2048:
                print(cnt)
        print(dataset[0]['inst'])
        results = pipe(
            KeyDataset(dataset, 'inst'),
            generation_config=gen_conf,
        )

        out_file = join(args.result_dir, args.output_name)
        os.makedirs(dirname(out_file), exist_ok=True)
        mode = 'a' if args.append else 'w'
        with open(out_file, mode, encoding='utf-8') as fp:
            fp.writelines(jsoned(results, dataset))
        print(f"write to {out_file}")

    def humaneval(self):
        def generate_inst_(inst_type, **data):
            desc = data['prompt'].partition('"""')[2].partition('"""')[0].strip()
            return humaneval_inst[inst_type].format(
                desc=desc, **data
            )

        def res2json(results, dataset):
            results_ = tqdm(results, desc='generating', total=len(dataset))
            for idx, res in enumerate(results_):
                task_id = dataset[idx]['task_id']
                # 一个res对应一个task，有多个生成样本
                for sample in res:
                    sample_dict = {
                        'task_id': task_id,
                        'completion': sample['generated_text']
                    }
                    yield json.dumps(sample_dict, ensure_ascii=False)+'\n'

        generate_inst = partial(generate_inst_, inst_type=self.args.inst_type)
        dataset = list(read_problems().values())
        self.gen(dataset, generate_inst, res2json)

    def codescope(self):
        def generate_inst_(inst_type, **data):
            lang = data['lang_cluster'].lower()
            n3 = '\n'*3
            io = [
                f"Input{n3}{i}{n3}Output{n3}{o}"
                for i, o in zip(data['sample_inputs'], data['sample_outputs'])
            ]
            sampleio = n3.join(io)
            return codescope_inst[inst_type].format(
                lang=lang, sampleio=sampleio, **data
            )

        def res2json(results, dataset):
            results_ = tqdm(results, desc='generating', total=len(dataset))
            for res, data in zip(results_, dataset):
                for sample in res:
                    sample_dict = {
                        "src_uid": data['src_uid'],
                        "id": data['id'],
                        "lang_cluster": data['lang_cluster'].lower(),
                        "lang": env_map[data['lang_cluster'].lower()],
                        "difficulty": data['difficulty'],
                        "testcases": data['testcases'],
                        "source_code": sample['generated_text'],
                    }
                    yield json.dumps(sample_dict, ensure_ascii=False)+'\n'

        def res2json_with_process(results, dataset):
            results_ = tqdm(results, desc='generating', total=len(dataset))
            for res, data in zip(results_, dataset):
                for sample in res:
                    lang = data['lang_cluster'].lower()
                    src = (
                        sample['generated_text']
                        .partition(f'[{lang}]')[2]
                        .partition(f'[/{lang}]')[0]
                    )
                    sample_dict = {
                        "src_uid": data['src_uid'],
                        "id": data['id'],
                        "lang_cluster": data['lang_cluster'].lower(),
                        "lang": env_map[data['lang_cluster'].lower()],
                        "difficulty": data['difficulty'],
                        "testcases": data['testcases'],
                        "source_code": src,
                    }
                    yield json.dumps(sample_dict, ensure_ascii=False)+'\n'

        generate_inst = partial(generate_inst_, inst_type=self.args.inst_type)
        dataset = load_dataset('json', data_files=self.args.datapath, split='train').to_list()
        self.gen(
            dataset, generate_inst,
            res2json if not self.args.use_process else res2json_with_process
        )

    def mbpp(self):
        pass


if __name__ == "__main__":
    parser = HfArgumentParser(GenArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()

    generator = Generator(eval_args)
    if eval_args.task == 'humaneval':
        generator.humaneval()
    elif eval_args.task == 'codescope':
        generator.codescope()
