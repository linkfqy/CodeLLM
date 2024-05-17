from textstat import flesch_reading_ease
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from tqdm import tqdm

lang_id = {
    "UNKNOWN_LANGUAGE": 0,
    "Python2": 1,
    "C++": 2,
    "Python3": 3,
    "Java": 4,
}
lang_list = ["UNKNOWN_LANGUAGE", "Python2", "C++", "Python3", "Java"]
lang_short = {
    None: '',
    "C++": 'cpp',
    "Python3": 'py',
    "Java": 'java',
}

alpaca_input = '''As an expert code developer with years of experience, please provide the source code based on the problem description. The detailed information are as follows:
1. Problem description: {desc}
2. Programming language: {lang}
'''


def makedata(
    split, datatype, maxsol,
    langfilter=None, maxdiff=3500, minease=-999, max_token=2000, no_image=False
):
    def count_token(s):
        tokens = tokenizer(s)['input_ids']
        return len(tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        "model/CodeLlama-7b-Instruct-hf",
        model_max_length=int(1e30)
    )

    dataset = load_dataset("./data/code_contests", split=split)
    outname = f"{split}_{datatype}{max_token}_sol{maxsol}{lang_short[langfilter]}"
    outname += f"ease{minease}" if minease != -999 else ''

    f = open(f"data/CCSFT/{'noimage/'*int(no_image)}{outname}.jsonl", 'w')
    for data in tqdm(dataset, 'generating'):
        desc = data['description']
        if len(desc) < 100 or "This is an interactive problem" in desc:
            continue
        if no_image and "<image>" in desc:
            continue
        ease = flesch_reading_ease(desc)
        if ease < minease:
            continue
        cnt = 0
        languages = data['solutions']['language']
        solutions = data['solutions']['solution']
        for langid, sol in zip(languages, solutions):
            langname = lang_list[langid]
            if langname == 'Python2':
                continue
            if langfilter is not None and langfilter != langname:
                continue
            if len(desc)+len(sol) > 2.7*max_token or count_token(desc+sol) > max_token:
                continue
            if datatype == 'alpaca':
                newdata = {
                    'name': data['name'],
                    'instruction': alpaca_input.format(desc=desc, lang=langname),
                    'output': sol,
                }
            else:
                newdata = {
                    'name': data['name'],
                    'description': desc,
                    'language': langname,
                    'solution': sol,
                }
            f.write(json.dumps(newdata, ensure_ascii=False)+'\n')
            cnt += 1
            if cnt >= maxsol:
                break
    f.close()


if __name__ == "__main__":
    makedata('train', 'alpaca', 10, langfilter='Python3', max_token=1980, no_image=True)
    makedata('train', 'alpaca', 10, langfilter='C++', max_token=1980, no_image=True)
    makedata('train', 'alpaca', 10, langfilter='Java', max_token=1980, no_image=True)
    makedata('validation', 'alpaca', 10, langfilter='Python3', max_token=1980, no_image=True)
    makedata('validation', 'alpaca', 10, langfilter='C++', max_token=1980, no_image=True)
    makedata('validation', 'alpaca', 10, langfilter='Java', max_token=1980, no_image=True)
