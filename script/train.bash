name=codepy_1

deepspeed \
    --include localhost:1,2,3,4,6,7 \
    src/train.py \
    --model_name_or_path model/CodeLlama-7b-Instruct-hf \
    --train_data_path data/CCSFT/train_alpaca950_sol10onlypy.jsonl \
    --valid_data_path data/CCSFT/validation_alpaca950_sol10onlypy.jsonl \
    --output_dir save/${name} \
    --num_train_epochs 2 \
    --model_max_length 2048 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --report_to "wandb" \
    --run_name ${name} \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --deepspeed src/configs/deepspeed_config.json \
    --bf16 True \
    --prompt_type "native"
    # --neftune_noise_alpha 5
