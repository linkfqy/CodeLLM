from datasets import load_dataset
import json


def make3lang():
    langs = ('C++', 'Java', 'Python')
    dataset = load_dataset('json', data_files='data/program_synthesis_data.jsonl', split='train')
    dataset.filter(
        lambda data: data['lang_cluster'] in langs
    ).to_json('data/cs_3lang.jsonl')


def make_unique(lang, langname):
    dataset = load_dataset('json', data_files='data/program_synthesis_data.jsonl', split='train')
    used_uids = set()
    res = []
    for data in dataset:
        if data['src_uid'] in used_uids:
            continue
        used_uids.add(data['src_uid'])
        data['lang_cluster'] = lang
        res.append(json.dumps(data)+'\n')
    open(f'data/cs_unique_{langname}.jsonl', 'w').writelines(res)


if __name__ == '__main__':
    # make3lang()
    # make_unique('Python','py')
    make_unique('Java', 'java')
    make_unique('C++', 'cpp')
