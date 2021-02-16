"""
Single graph is described by an instanc eof torch_geometric.data.Data
which hold the following attrs by default:
* data.x: Node feature matrix with shape [num_nodes, num_node_features]
* data.edge_index: Graph connectivity in COO format with shape [2, num_edges] <edge_list>
* data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
* data.y: Target to train against (arbitrary shape)
* data.pos: Node position matrix with shape [num_nodes, num_dimensions]
"""

import torch
from torch_geometric.data import Data

# connectivity matrix describing: X_1 <--> X_2 <--> X_3A
# [0, 1, 1, 2] => 0 connects to 1, 1 connects to 2
# [1, 0, 2, 1] => 1 connects to 0, 2 connects to 1
# accounting for both directions of edge (removing directionality)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# Node feature vectors
x = torch.tensor([[-1], [0], [1]])

data = Data(x=x, edge_index=edge_index)
print(data)

# NOTE: Can also write edges as list of tuples (src, dst)
# but need to transpose (creates same lists as those used above)
# and call contiguous (join lists into two lists) on it
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data2 = Data(x=x, edge_index=edge_index.t().contiguous())
print(data2)
