from textstat import flesch_reading_ease
import json
from matplotlib import pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def count_token(tokenizer, seq):
    res = tokenizer(seq, return_tensors='pt')
    return res['input_ids'].shape[1]


def ease_hist():
    plt.cla()
    ease_arr = []
    with open("/fqyVol/CodeScope/data/program_synthesis_data.jsonl") as f:
        while line := f.readline():
            sample = json.loads(line)
            ease = flesch_reading_ease(sample["description"])
            # if ease < 60:
            #     print(sample['id'])
            #     print(sample["description"])
            #     print('-'*20)
            ease_arr.append(ease)
    plt.hist(ease_arr, bins=50)
    plt.tight_layout()
    plt.savefig("image/ease_hist.png", dpi=500)
    print(max(ease_arr), min(ease_arr))


def sol_hist():
    plt.cla()
    dataset = load_dataset("data/code_contests", split="train")
    print(f"problem num: {len(dataset)}")
    sol_arr = []
    cnt = 0
    for data in dataset:
        sol = len(data['solutions']['solution'])
        cnt += min(5, sol)
        sol_arr.append(sol)
    print(cnt)
    plt.hist(sol_arr, bins=50, log=True)
    plt.tight_layout()
    plt.savefig("image/sol_hist.png", dpi=500)
    print(max(sol_arr), min(sol_arr))

def diff_hist():
    plt.cla()
    dataset = load_dataset("data/code_contests", split="train")
    print(f"problem num: {len(dataset)}")
    diff_arr = []
    cnt = 0
    for data in dataset:
        sol = len(data['solutions']['solution'])
        cnt += min(5, sol)
        if data['source']==2:
            diff_arr.append(data['cf_rating'])
    print(cnt)
    plt.hist(diff_arr,range=(800,3500), bins=27)
    plt.tight_layout()
    plt.savefig("image/diff_hist.png", dpi=500)
    print(max(diff_arr), min(diff_arr))
    print(sum(diff for diff in diff_arr if diff<=2800)/sum(diff_arr))


def lang_pie():
    plt.cla()
    dataset = load_dataset("data/code_contests", split="train")
    lang_arr = [0]*5
    for data in dataset:
        langs = data['solutions']['language']
        for lang in langs:
            lang_arr[lang] += 1
    print(f"sol num:{sum(lang_arr)}")
    plt.pie(lang_arr, labels=["UNKNOWN_LANGUAGE", "PYTHON2", "CPP", "PYTHON3", "JAVA"])
    plt.tight_layout()
    plt.savefig("image/lang_pie.png", dpi=500)
    print(lang_arr)
    print([x/sum(lang_arr) for x in lang_arr])


def token_hist():
    plt.cla()
    # len:token=2.7:1
    tokenizer = AutoTokenizer.from_pretrained(
        "model/CodeLlama-7b-Instruct-hf",
        model_max_length=16384,
        use_fast=True,
    )
    dataset = load_dataset("data/code_contests", split="train")
    len_arr = []
    for data in tqdm(dataset):
        desc_len = len(data['description'])
        sol_list = data['solutions']['solution']
        for sol in sol_list:
            sol_len = len(sol)
            len_arr.append(desc_len+sol_len)
    plt.hist(len_arr, bins=100, range=(0, 6000))
    plt.savefig("image/token_hist.png", dpi=500)
    print(max(len_arr), min(len_arr))


def loss_plot(trainer_state):
    plt.cla()
    with open(trainer_state, 'r') as f:
        state = json.load(f)
    epoch, loss = [],[]
    for log in state['log_history']:
        if 'loss' in log.keys():
            epoch.append(log['epoch'])
            loss.append(log['loss'])
    plt.plot(epoch, loss)
    plt.savefig("image/loss_plot.png", dpi=500)


if __name__ == '__main__':
    # loss_plot("save/codev5/trainer_state.json")
    # sol_hist()
    # token_hist()
    diff_hist()
    # ease_hist()
    # lang_pie()
