# TransE

Implementation of TransE model based on mxnet.gluon

Paper: [Translating embeddings for modeling multi-relational data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

Paper:

|           | WN18(RAW) | WN18(Filt) |
| --------- | --------- | ---------- |
| Mean-Rank | 263       | 251      |
| Hit@10    | 0.754     | 0.892      |

Our result:

|           | WN18(RAW) | WN18(Filt) |
| --------- | --------- | ---------- |
| **Mean-Rank** | **240.0**     | **235.7**    |
| Hit@1     | 0.1558    | 0.12     |
| **Hit@10**   | **0.7496**    | **0.7898**      |
| Hit@20    | 0.8194    | 0.8548     |
| Hit@50    | 0.9126    | 0.909     |
| Hit@100   | 0.9776    | 0.9368     |
