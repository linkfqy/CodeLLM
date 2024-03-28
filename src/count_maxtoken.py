from datasets import load_dataset
from transformers import AutoTokenizer

jsonl_list = [
    "./data/humaneval-x/humaneval_python.jsonl",
    "./data/humaneval-x/humaneval_java.jsonl",
    "./data/humaneval-x/humaneval_js.jsonl",
    "./data/humaneval-x/humaneval_cpp.jsonl",
    "./data/humaneval-x/humaneval_go.jsonl",
    "./data/humaneval-x/humaneval_rust.jsonl",
]

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/home/fqy/HFmodels/CodeLlama-7b-Instruct-hf")
    for jsonl in jsonl_list:
        dataset = load_dataset('json', data_files={'test': jsonl})['test']
        maxtoken, taskid = 0, 0
        for data in dataset:
            input = tokenizer(data['canonical_solution'], return_tensors='pt')
            length = input['input_ids'].shape[1]
            if length > maxtoken:
                maxtoken, taskid = length, data['task_id']
        print(f"{jsonl}: {maxtoken} tokens at {taskid}")
