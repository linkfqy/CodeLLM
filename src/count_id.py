from datasets import load_dataset


def count_id(file):
    dataset = load_dataset('json', data_files=file, split='train')
    s = set(data['name'] for data in dataset)
    print(f"{file}: {len(s)} problems")


if __name__ == '__main__':
    count_id("data/CCSFT/noimage/train_alpaca950_sol10py.jsonl")
    count_id("data/CCSFT/train_alpaca950_sol10onlypy.jsonl") # codev4_2
    count_id("data/CCSFT/train_alpaca_sol10onlypy.jsonl") # codev4
    count_id("data/CCSFT/train_alpaca950_sol10nopy2.jsonl") # codev5_2
    count_id("data/CCSFT/train_alpaca_sol5nopy2.jsonl")