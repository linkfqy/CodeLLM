name=codev4

deepspeed \
    --include localhost:0,1,2,3,4,5 \
    src/train.py \
    --model_name_or_path model/CodeLlama-7b-Instruct-hf \
    --train_data_path data/train_alpaca_sol10onlypy.jsonl \
    --valid_data_path data/validation_alpaca_sol10onlypy.jsonl \
    --output_dir save/${name} \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --report_to "wandb" \
    --run_name ${name} \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --deepspeed src/configs/deepspeed_config.json \
    --bf16 True
