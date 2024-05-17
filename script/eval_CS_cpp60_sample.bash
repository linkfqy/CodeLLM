export CUDA_VISIBLE_DEVICES=7

temp=0.8
top_k=50
top_p=0.95
candidate_num=5
max_new_tokens=2048  # 和具体任务有关
m_name=codev5_2
result_dir=./result/${m_name}
inst_type=native_cf

output_name=CS_cpp60_${inst_type}T${temp}N${candidate_num}.jsonl
output_path=${result_dir}/${output_name}

python ./src/generate.py  \
    --task codescope  \
    --datapath data/cs_cpp60.jsonl  \
    --ckpt save/${m_name}/checkpoint-final  \
    --tokenizer save/${m_name}/checkpoint-final  \
    --batch_size 2  \
    --result_dir ${result_dir}  \
    --output_name ${output_name}  \
    --inst_type ${inst_type}  \
    --temprature ${temp}  \
    --candidate_num ${candidate_num}  \
    --max_new_tokens ${max_new_tokens}  \
    --top_k ${top_k}  \
    --top_p ${top_p}  \
    # --repetition_penalty 1.2

python src/exec_eval/run_execeval.py \
    --code_path ${output_path} \
    --output_path ${result_dir}/runres/${output_name}

python src/cs_score.py \
    --res_path ${result_dir}/runres/${output_name}