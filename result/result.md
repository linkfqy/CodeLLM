| 文件                                   | HumanEval         | 备注          |
| -------------------------------------- | ----------------- | ------------- |
| cl7b/best/HE_greedyN1.proc.jsonl       | 'pass@1': 0.421   | ins1,len=512  |
| cl7b/best/backup/HE_T0.8N10.proc.jsonl | 'pass@10': 0.701  | ins1,len=512  |
| cl7b/best/HE_T0.8N100.proc.jsonl       | 'pass@100': 0.896 | ins2,len=512  |
| cl7b/best/HE_greedynewcfN1.jsonl       | 'pass@1': 0.433   | newcf,len=512 |

| model         | c++ | java | py  | 备注 |
| ------------- | --- | ---- | --- | ---- |
| code4(E)      |     |      | 5   |      |
| code4(H)      |     |      | 0   |      |
| code5_2(E)    | 8   | 2    | 1   |      |
| code5_2(H)    | 0   | 0    | 0   |      |
| code5_2(E)    | 7   | 5    | 2   |      |
| code5_2(H)    | 0   | 0    | 0   |      |
| codecpp_2(E)  | 9   |      |     |      |
| codecpp_2(H)  | 0   |      |     |      |
| codejava_1(E) |     | 10   |     |      |
| codejava_1(H) |     | 0    |     |      |
| cl7b(E)       | 1   | 1    | 0   |      |
| cl7b(H)       | 0   | 0    | 0   |      |