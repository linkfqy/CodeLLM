export CUDA_VISIBLE_DEVICES=2,3

temp=0.8
candidate_num=10
max_new_tokens=512  # 和具体任务有关
result_dir=./result/original

output_name=HE_T${temp}N${candidate_num}.jsonl
# output_name=HE_greedyN${candidate_num}.jsonl
output_path=${result_dir}/${output_name}

python ./src/evaluate.py  \
    --eval_type humaneval  \
    --ckpt codellama/CodeLlama-7b-Instruct-hf  \
    --tokenizer codellama/CodeLlama-7b-Instruct-hf  \
    --batch_size 8  \
    --result_dir ${result_dir}  \
    --output_name ${output_name}  \
    --temprature ${temp}  \
    --candidate_num ${candidate_num}  \
    --max_new_tokens ${max_new_tokens}  \
    # --greedy_encode

python ./src/process_humaneval.py --input_file ${output_path}
