export CUDA_VISIBLE_DEVICES=7

temp=0.0
candidate_num=1
max_new_tokens=512  # 和具体任务有关
m_name=cl7b
result_dir=./result/${m_name}
inst_type=newcf

# output_name=HE_T${temp}N${candidate_num}.jsonl
output_name=HE_greedy${inst_type}N${candidate_num}.jsonl
output_path=${result_dir}/${output_name}

python ./src/generate.py  \
    --task humaneval  \
    --ckpt codellama/CodeLlama-7b-Instruct-hf  \
    --tokenizer codellama/CodeLlama-7b-Instruct-hf  \
    --batch_size 8  \
    --result_dir ${result_dir}  \
    --output_name ${output_name}  \
    --inst_type ${inst_type}  \
    --temprature ${temp}  \
    --candidate_num ${candidate_num}  \
    --max_new_tokens ${max_new_tokens}  \
    --greedy_decode  \
    # --repetition_penalty 1.1

# python ./src/process_humaneval.py --input_file ${output_path}

evaluate_functional_correctness ${output_path}