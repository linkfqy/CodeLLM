High Efficiency: Training 7B model with 50k examples/epoch & batch_size=64 within 1 hour on 8 x V100 GPUs
| LLaMA | Batch Size | V100s | Time (h) |
| ----- | ---------- | ----- | -------- |
| 7B    | 64         | 8     | 1.00     |

1. *筛选`deepmind/code_contests`：每题最多10/5解，去除py2
2. *跑通LlamaX-deepspeed
3. *配CodeScope环境
4. cosine lr按epoch分布（不必）
5. 训练集shuffle
6. *用onlypy数据集训练，统计token
7. 可能的参数：2epoch、
8. 原生指令模式微调
9. vllm等方法加速推理
10. 使用pipeline重写CodeScope生成
11. 测试：MBPP、APPS、HumanEval+、自建cf数据集

对于50k example * 3 epoch:
8*A800 batch96 8.3h

对于100k example * 3 epoch 2048token：
8*A800 batch 128 15.25h


Llama 1 supports up to 2048 tokens, Llama 2 up to 4096, CodeLlama up to 16384.

```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
```
