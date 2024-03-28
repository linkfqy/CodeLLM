def prompt_and_tokenize(data: dict, tokenizer):
    # data['args'] = data['returns'] = data['raises'] = ''
    # for param in data['docstring_params']['params']:
    #     data['args'] += param['identifier']+(' ' if not param.has_key('type') else param['type']) + param['docstring']
    prompt = '''You are a powerful {language} code model. Your job is to complete {language} code from the document.

### Document:
function name: {identifier}
description:
{docstring}

### Code:
{original_string}
'''.format_map(data)
    res = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None,
    )
    res["labels"] = res["input_ids"].copy()
    return res