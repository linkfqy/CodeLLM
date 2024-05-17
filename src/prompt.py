humaneval_inst = {
    'v1': (
        'Below is an instruction that describes a task. Write a '
        'response that appropriately completes the request.\n'
        '\n'
        '\n'
        '### Instruction:\n'
        'Complete this Python function in a markdown code block '
        'to solve programming problem:\n'
        '```python\n'
        '{prompt}\n'
        '```\n'
        '\n'
        '### Response:'
    ),
    '500': (
        'Below is an instruction that describes a task. Write a '
        'response that appropriately completes the request.\n'
        '\n'
        '\n'
        '### Instruction:\n'
        'Complete this Python function in a markdown code block '
        'to solve programming problem:\n'
        '```python\n'
        '{prompt}\n'
        '```\n'
        '\n'
        '### Response:'
    ),
    'cf': (
        'Below is an instruction that describes a task. Write a '
        'response that appropriately completes the request.\n'
        '\n'
        '\n'
        '### Instruction:\n'
        'As an expert code developer with years of experience, '
        'please provide the source code based on the problem '
        'description. The detailed information are as follows:\n'
        '1. Problem description: {desc}\n'
        '2. Programming language: Python\n'
        '\n'
        '### Response:\n'
        '{prompt}'
    ),
    'newcf': (
        '[INST]<<SYS>>Below is an instruction that describes a '
        'task. Write a response that appropriately completes the '
        'request.\n'
        '<</SYS>>As an expert code developer with years of '
        'experience, please provide the source code based on the '
        'problem description. The detailed information are as '
        'follows:\n'
        '1. Problem description: {desc}\n'
        '2. Programming language: Python\n'
        '[/INST]{prompt}'
    ),
    'no_prompt': "{prompt}",
}

codescope_inst = {
    'native': (
        '[INST]<<SYS>>Below is an instruction that describes a '
        'task. Write a response that appropriately completes the '
        'request.\n'
        '<</SYS>>As an expert code developer with years of '
        'experience, please provide the source code based on the '
        'problem description. The detailed information are as '
        'follows:\n'
        "1. Problem description: {description}\n"
        "2. Input specification: {input_specification}\n"
        "3. Output specification: {output_specification}\n"
        "4. Sample inputs: {sample_inputs}\n"
        "5. Sample outputs: {sample_outputs}\n"
        "6. Sample explanations: {notes}\n"
        "7. Programming language: {lang} \n"
        "[/INST]"
    ),
    'native_cf': (
        '[INST]<<SYS>>Below is an instruction that describes a '
        'task. Write a response that appropriately completes the '
        'request.\n'
        '<</SYS>>As an expert code developer with years of '
        'experience, please provide the source code based on the '
        'problem description. The detailed information are as '
        'follows:\n'
        "1. Problem description: {description}\n"
        "2. Input specification: {input_specification}\n"
        "3. Output specification: {output_specification}\n"
        "4. Sample inputs and outputs: {sampleio}\n"
        "5. Sample explanations: {notes}\n"
        "6. Programming language: {lang} \n"
        "[/INST]"
    ),
    'native_cf_block': (
        '[INST]<<SYS>>Below is an instruction that describes a '
        'task. Write a response that appropriately completes the '
        'request.\n'
        '<</SYS>>As an expert code developer with years of '
        'experience, please provide the source code based on the '
        'problem description. The detailed information are as '
        'follows:\n'
        "1. Problem description: {description}\n"
        "2. Input specification: {input_specification}\n"
        "3. Output specification: {output_specification}\n"
        "4. Sample inputs and outputs: {sampleio}\n"
        "5. Sample explanations: {notes}\n"
        "6. Programming language: {lang} \n"
        "Your code must be between [{lang}] and [/{lang}] tags.\n"
        "[/INST]"
    ),
    'alpaca_cf': (
        'Below is an instruction that describes a '
        'task. Write a response that appropriately completes the '
        'request.\n'
        '### Instruction:\n'
        'As an expert code developer with years of '
        'experience, please provide the source code based on the '
        'problem description. The detailed information are as '
        'follows:\n'
        "1. Problem description: {description}\n"
        "2. Input specification: {input_specification}\n"
        "3. Output specification: {output_specification}\n"
        "4. Sample inputs and outputs: {sampleio}\n"
        "5. Sample explanations: {notes}\n"
        "6. Programming language: {lang} \n"
        "### Response:\n"
    ),
}
