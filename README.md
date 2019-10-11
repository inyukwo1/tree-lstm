# Tree LSTM
This repository contains a Pytorch Implementation of "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks "
(https://arxiv.org/abs/1503.00075).

This contains two type of tree-lstm (Child sum, N-ary). This was tested by Python 3.6, Pytorch 1.3.0., and this internally uses dgl 0.4.0


This repository referenced https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py

Installation
===
```pip install tree-lstm```

after installed, you can use this via

```import TreeLSTM, Tree, BatchedTree```

Usage
=====
1. make Tree object (initialize with arbitrary tensor)
2. make BatchedTree object using list of Tree object
3. make TreeLSTM object (inherited from torch.nn.Module)
4. pass BatchedTree object into TreeLSTM object

For more detailed usage, please see `test.py`


