from tree_lstm import TreeLSTM, Tree, BatchedTree
from random import randint, choice
import torch


X_SIZE = 30
H_SIZE = 30


def make_random_tree():
    tree = Tree(H_SIZE)
    max_depth = randint(1, 4)

    # root
    root_id = tree.add_node(parent_id=None, tensor=torch.rand(X_SIZE))
    parents = [root_id]
    for _ in range(max_depth):
        child_num = randint(1, 4)
        new_parents = []
        for _ in range(child_num):
            parent_id = choice(parents)
            child_id = tree.add_node(parent_id=parent_id, tensor=torch.rand(X_SIZE))
            new_parents.append(child_id)
        parents = new_parents
    return tree


if __name__ == "__main__":
    tree_list = []
    for _ in range(5):
        tree = make_random_tree()
        tree_list.append(tree)
    batched_tree = BatchedTree(tree_list)
    model = TreeLSTM(x_size=X_SIZE, h_size=H_SIZE, dropout=0.3, cell_type='n_ary', n_ary=4)
    out_batch = model(batched_tree)
    print(out_batch.get_hidden_state())