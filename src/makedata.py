from textstat import flesch_reading_ease
from datasets import load_dataset
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

alpaca_input = '''As an expert code developer with years of experience, please provide the source code based on the problem description. The detailed information are as follows:
1. Problem description: {desc}
2. Programming language: {lang}
'''


def makedata(split, datatype, maxsol, nopy2=True, onlypy=False, minease=-999, maxlen=int(2.7*2000)):
    dataset = load_dataset("./data/code_contests", split=split)
    outname = f"{split}_{datatype}_sol{maxsol}{'nopy2'*int(nopy2)}{'onlypy'*int(onlypy)}"
    outname += f"ease{minease}" if minease != -999 else ''

    f = open(f"data/{outname}.jsonl", 'w')
    for data in tqdm(dataset, 'generating'):
        desc = data['description']
        if len(desc) < 100 or "This is an interactive problem" in desc:
            continue
        ease = flesch_reading_ease(desc)
        if ease < minease:
            continue
        cnt = 0
        languages = data['solutions']['language']
        solutions = data['solutions']['solution']
        for langid, sol in zip(languages, solutions):
            langname = lang_list[langid]
            if (nopy2 and langid == 1) or (onlypy and langid != 3):
                continue
            if len(desc)+len(sol) > maxlen:
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
    makedata('train', 'alpaca', 5)
    makedata('validation', 'alpaca', 5)
    makedata('train', 'alpaca', 10)
    makedata('validation', 'alpaca', 10)
    makedata('train', 'alpaca', 10, nopy2=False, onlypy=True)
    makedata('validation', 'alpaca', 10, nopy2=False, onlypy=True)
