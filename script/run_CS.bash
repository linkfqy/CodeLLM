result_dir=./result/codev6
output_name=CS_3lang_native_cfT0.8N5.jsonl

output_path=${result_dir}/${output_name}
#LLM输出的output_path

python src/exec_eval/run_execeval.py \
    --code_path ${output_path} \
    --output_path ${result_dir}/runres/${output_name}

python src/cs_score.py \
    --res_path ${result_dir}/runres/${output_name}