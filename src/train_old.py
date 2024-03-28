import torch
import os
from datetime import datetime
from typing import List
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from dataclasses import dataclass, field
from prompt import prompt_and_tokenize


@dataclass
class ModelArguments:
    model_path: str


@dataclass
class DataArguments:
    data_dir: str
    data_split: str = 'small'
    data_languages: List[str] = field(default_factory=list)


@dataclass
class TrainArguments:
    batch_size: int = 128
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = field(init=False)
    per_device_eval_batch_size: int = field(init=False)
    eval_accumulation_steps: int = field(init=False)
    epochs: float = 3.0
    learning_rate: float = 3e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    output_dir: str = "./save"
    debug: bool = False

    def __post_init__(self):
        # // torch.cuda.device_count()
        self.gradient_accumulation_steps = self.batch_size // self.per_device_train_batch_size
        self.per_device_eval_batch_size = self.per_device_train_batch_size*4
        self.eval_accumulation_steps = self.gradient_accumulation_steps//2
        self.time = datetime.now().strftime('%Y%m%d-%H%M')
        self.output_dir = self.output_dir+f"/{self.time}"


def init_dataset(data_dir, split_set: str, languages: list, tokenizer):
    dataset = load_dataset(
        data_dir,
        data_dir=data_dir,
        split_set=[split_set],
        languages=languages,
        # streaming=True,
    )
    dataset = dataset[split_set.replace('/', '_')]
    return dataset


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    # model_args: ModelArguments
    # data_args: DataArguments
    # train_args: TrainArguments

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype='auto',
        device_map='auto',
    )
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    train_dataset = init_dataset(
        data_args.data_dir,
        f"train/{data_args.data_split}" if not train_args.debug else "test",
        data_args.data_languages,
        tokenizer
    )
    if train_args.debug:
        val_dataset = train_dataset.select(range(100))
    else:
        val_dataset = init_dataset(
            data_args.data_dir,
            "validation",
            data_args.data_languages,
            tokenizer
        )
    train_dataset = train_dataset.map(
        lambda d: prompt_and_tokenize(d, tokenizer))
    val_dataset = val_dataset.map(lambda d: prompt_and_tokenize(d, tokenizer))
    training_args = TrainingArguments(
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        # eval_accumulation_steps=train_args.eval_accumulation_steps,
        warmup_ratio=0.1,
        # max_steps=400,
        num_train_epochs=train_args.epochs,
        learning_rate=train_args.learning_rate,
        optim="adamw_torch",
        adam_beta1=train_args.adam_beta1,
        adam_beta2=train_args.adam_beta2,
        group_by_length=True,
        # fp16=True,
        # fp16_full_eval=True,
        # log and save strategy default to "steps"
        evaluation_strategy="no",  # "steps",
        # logging_steps=20,
        # eval_steps=20,
        # save_steps=20,
        output_dir=train_args.output_dir,
        save_total_limit=5,
        # load_best_model_at_end=True,
        # metric_for_best_model="loss",
        # greater_is_better=False,
        # wandb options
        report_to="wandb",
        run_name=f"codellama-{train_args.time}",
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )
    # model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    train()
