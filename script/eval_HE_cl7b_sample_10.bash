export CUDA_VISIBLE_DEVICES=7

temp=0.6
top_k=50
top_p=0.95
candidate_num=10
max_new_tokens=512  # 和具体任务有关
m_name=cl7b
result_dir=./result/${m_name}
inst_type=newcf

output_name=HE_${inst_type}T${temp}N${candidate_num}.jsonl
output_path=${result_dir}/${output_name}

python ./src/generate.py  \
    --task humaneval  \
    --ckpt codellama/CodeLlama-7b-Instruct-hf  \
    --tokenizer codellama/CodeLlama-7b-Instruct-hf  \
    --batch_size 4  \
    --result_dir ${result_dir}  \
    --output_name ${output_name}  \
    --inst_type ${inst_type}  \
    --temprature ${temp}  \
    --candidate_num ${candidate_num}  \
    --max_new_tokens ${max_new_tokens}  \
    --top_k ${top_k}  \
    --top_p ${top_p}  \
    # --repetition_penalty 1.2

evaluate_functional_correctness ${output_path}
