from collections import Counter
from datasets import load_dataset
from api_comm import APICommunication, ExtendedUnittest
import argparse


def add_exec_outcome(example):
    language = example['lang']
    source_code = example['source_code']
    hidden_unit_tests = eval(example['testcases'])

    unittests = []
    for hidden_unit_test in hidden_unit_tests:
        unittests.append(
            ExtendedUnittest(
                input=hidden_unit_test['input'],
                output=hidden_unit_test['output']
            ).json()
        )

    api = APICommunication()
    response = api.execute_code(
        language=language,
        source_code=source_code,
        unittests=unittests
    )
    print(example['id'], response)

    example['exec_outcome'] = response[0]

    return example


def main(code_path, output_path):
    dataset = load_dataset('json', split='train', data_files=code_path)
    dataset = dataset.map(add_exec_outcome)

    lang_counts = Counter(dataset['lang'])
    for lang, count in lang_counts.items():
        print(f'{lang}: {count}')

    lang_cluster_counts = Counter(dataset['lang_cluster'])
    for lang_cluster, count in lang_cluster_counts.items():
        print(f'{lang_cluster}: {count}')

    dataset.to_json(output_path, lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', type=str,)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    main(args.code_path, args.output_path)
