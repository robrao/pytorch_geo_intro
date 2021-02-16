from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

print("Enzyme Dataset")
print(f"Dataset size: {len(dataset)}")
print(f"Num classes in dataset: {dataset.num_classes}")
print(f"Num node features in dataset: {dataset.num_node_features}")

first_graph = dataset[0]
print(f"First Graph: {first_graph}")
print(f"Graph is undirected: {first_graph.is_undirected()}")

# 90/10 split
train_dataset = dataset[:540]
test_dataset = dataset[540:]

dataset2 = Planetoid(root='/tmp/Cora', name='Cora')
print("Cora Dataset")
print(f"Num of graphs: {len(dataset2)}")

data = dataset2[0]
# Denotes against which nodes to train
print(f"Total training nodes: {data.train_mask.sum().item()}")
# Denotes against which nodes to validate (early stopping)
print(f"Total validation nodes: {data.val_mask.sum().item()}")
# Denotes aginst which nodes to test
print(f"Total test nodes: {data.test_mask.sum().item()}")