export CUDA_VISIBLE_DEVICES=6,7

temp=0.0
candidate_num=1
result_dir=./result/original

# output_name=HE_T${temp}N${candidate_num}_noprompt.jsonl
output_name=HE_greedyN${candidate_num}_noprompt.jsonl
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
    --greedy_encode  \
    --no_prompt
