export CUDA_VISIBLE_DEVICES=6,7

temp=0.0
candidate_num=1
max_new_tokens=512  # 和具体任务有关
result_dir=./result/codev1

# output_name=HE_T${temp}N${candidate_num}.jsonl
output_name=HE_greedycfN${candidate_num}.jsonl
output_path=${result_dir}/${output_name}

python ./src/evaluate.py  \
    --eval_type humaneval  \
    --ckpt save/codev1/checkpoint-final  \
    --tokenizer save/codev1/checkpoint-final  \
    --batch_size 16  \
    --result_dir ${result_dir}  \
    --output_name ${output_name}  \
    --temprature ${temp}  \
    --candidate_num ${candidate_num}  \
    --max_new_tokens ${max_new_tokens}  \
    --inst_type cf  \
    --greedy_decode

# python ./src/process_humaneval.py --input_file ${output_path}
