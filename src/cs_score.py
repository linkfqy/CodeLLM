from datasets import load_dataset
import argparse
import pandas as pd
import numpy as np


def add_passed(example):
    example['passed'] = all(
        outcome['exec_outcome'] == 'PASSED'
        for outcome in example['exec_outcome']
    )
    return example


def score(res_path):
    dataset = load_dataset('json', data_files=res_path, split='train')
    dataset = dataset.map(add_passed)
    passkres = {}
    for data in dataset:
        if data['id'] not in passkres.keys():
            passkres[data['id']] = {
                'lang_cluster': data['lang_cluster'],
                'difficulty': data['difficulty'],
                'passed': False,
            }
        passkres[data['id']]['passed'] |= data['passed']

    langs = sorted(list(set(data['lang_cluster'] for data in dataset)))
    diff_dict = {
        'easy': (lambda x: 800 <= x < 1600),
        'hard': (lambda x: 1600 <= x < 2800),
    }
    psd = pd.DataFrame(
        np.zeros((len(diff_dict), len(langs))),
        columns=langs,
        index=['easy', 'hard']
    )
    num = psd.copy()
    for diff, diff_fn in diff_dict.items():
        for lang in langs:
            psd.loc[diff, lang] = sum(
                res['passed']
                for res in passkres.values()
                if diff_fn(res['difficulty']) and res['lang_cluster'] == lang
            )
            num.loc[diff, lang] = sum(
                1
                for res in passkres.values()
                if diff_fn(res['difficulty']) and res['lang_cluster'] == lang
            )
    pass_rate = psd/num
    pass_rate.to_csv(res_path + '.csv')
    print(psd, num, sep='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str,)
    args = parser.parse_args()

    score(args.res_path)
