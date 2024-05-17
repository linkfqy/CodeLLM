import pynvml
import time
import subprocess

Mi = 1024**2


def get_free_gpu():
    res = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if memory_info.used/Mi < 1000 and utilization.gpu == 0:
        # if memory_info.used/Mi < 50000 and utilization.gpu < 100:
            res.append(i)
    return res


task_list = [
    '''
export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
name=codev5_2

deepspeed \
    --include localhost:{gpus} \
    src/train.py \
    --model_name_or_path model/CodeLlama-7b-Instruct-hf \
    --train_data_path data/train_alpaca950_sol10nopy2.jsonl \
    --valid_data_path data/validation_alpaca950_sol10nopy2.jsonl \
    --output_dir save/${{name}} \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps {grad_acc} \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --report_to "wandb" \
    --run_name ${{name}} \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --deepspeed src/configs/deepspeed_config.json \
    --bf16 True \
    --neftune_noise_alpha 5
'''
]

if __name__ == "__main__":
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    for task in task_list:
        while True:
            gpus = get_free_gpu()
            if len(gpus)<4:
                time.sleep(5)
                continue
            break
        sh = task.format(
            gpus=str(gpus)[1:-1].replace(' ', ''),
            grad_acc=64//len(gpus)
        )
        with open("temp.sh","w") as f:
            f.write(sh)
        subprocess.call(['zsh','script/draft.sh'])
        

    pynvml.nvmlShutdown()
