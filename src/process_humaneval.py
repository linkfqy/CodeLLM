from human_eval.data import read_problems, write_jsonl, stream_jsonl
import os
from tqdm import tqdm
import argparse
import re


def process_responce(responce: str,idx):
    # return responce.partition('if __name__ == "__main__":')[0]
    completion = responce
    completion = completion.replace("\r", "")

    if "###" in completion:
        # print(completion)
        next_line = completion.index('###')
        completion = completion[:next_line].strip()
    
    # completion = re.sub(r'```.*\n', r'```\n', completion)
    if '```python' in completion:
        def_line = completion.index('```python')
        completion = completion[def_line+9:].strip()
        # print(completion)
        try:
            next_line = completion.index('```')
            completion = completion[:next_line].strip()
        except:
            print(f"Line {idx}:\n{completion}")
            print("================\n")
            return ""
        # print(completion)
    if '__name__ == "__main__"' in completion:
        next_line = completion.index('if __name__ == "__main__":')
        completion = completion[:next_line].strip()
        # print(completion)

    return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        '--input_file',
        type=str,
        help="")

    args = parser.parse_args()

    output = []
    for idx, sample in enumerate(stream_jsonl(args.input_file)):
        responce = sample['completion']
        sample['raw'] = responce
        sample['completion'] = process_responce(responce,idx+1)
        output.append(sample)

    root, ext = os.path.splitext(args.input_file)
    out_file = root+'.proc'+ext
    print(f"save to {out_file}")
    write_jsonl(out_file, output)
