# from pytorch3d.ops import GraphConv
import torch
import torch.nn as nn
from pytorch3d.ops.graph_conv import gather_scatter, gather_scatter_python
from pytorch3d.structures import list_to_padded


class LRFGraphConv(nn.Module):
    """A local reference frame graph convolution layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init: str = "normal",
    ):
        """Initializetion.

        Args:
            input_dim: Number of input features per vertex.
            output_dim: Number of output features per vertex.
            init: Weight initialization method. Can be one of ['zero', 'normal'].
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w0 = nn.Linear(input_dim, output_dim)

        if init == "normal":
            nn.init.normal_(self.w0.weight, mean=0, std=0.01)
            self.w0.bias.data.zero_()
        elif init == "zero":
            self.w0.weight.data.zero_()
        else:
            raise ValueError('Invalid LRFGraphConv initialization "%s"' % init)

    def forward(self, verts, edges, lrf=None):
        """Forward function.

        Args:
            verts: FloatTensor of shape (V, input_dim) where V is the number of
                vertices and input_dim is the number of input features
                per vertex. input_dim has to match the input_dim specified
                in __init__.
            edges: LongTensor of shape (E, 2) where E is the number of edges
                where each edge has the indices of the two vertices which
                form the edge.
            lrf: FloatTensor of shape (V, 3, 3) where V is the number of vertices.
                x, y, z axis are arranged accordingly at the last dimension.

        Returns:
            out: FloatTensor of shape (V, output_dim) where output_dim is the
                number of output features per vertex.
        """
        # print("_______________________________________-")
        if verts.is_cuda != edges.is_cuda:
            raise ValueError("verts and edges tensors must be on the same device.")
        if verts.shape[0] == 0:  # empty graph.
            return verts.new_zeros((0, self.output_dim)) * verts.sum()

        # Convert the edges to the padded tensor of index.
        adj_index, center_index = self.edged_to_adj_index(edges)  # V X max(N)
        # print(adj_index)
        # print(center_index)

        # Append (0, 0, 0) to the end of the vertices.
        end_point = torch.zeros_like(verts[-1:])
        padded_verts = torch.cat((verts, end_point), 0)
        # Get the relative points around the center.
        gather_neighbor = torch.index_select(padded_verts, dim=0, index=adj_index.view(-1))
        gather_center = torch.index_select(padded_verts, dim=0, index=center_index.view(-1))
        trans_neighbor = gather_neighbor - gather_center  # V * max(N) X 3
        # Get the shape back.
        trans_neighbor = trans_neighbor.view(verts.shape[0], -1, 3)  # V X max(N) X 3
        # Project the point onto the local reference frame.
        # print(trans_neighbor, lrf)
        rot_neighbor = torch.matmul(trans_neighbor, lrf)  # V X max(N) X 3
        verts_fea = self.w0(rot_neighbor)  # (V, max(N), output_dim)
        out = torch.sum(verts_fea, dim=1)  # (V, output_dim)

        # verts_w0 = self.w0(verts)  # (V, output_dim)
        # verts_w1 = self.w1(verts)  # (V, output_dim)

        # if torch.cuda.is_available() and verts.is_cuda and edges.is_cuda:
        #     neighbor_sums = gather_scatter(verts_w1, edges)
        #     print(neighbor_sums)
        # else:
        #     neighbor_sums = gather_scatter_python(
        #         verts_w1, edges
        #     )  # (V, output_dim)

        # Add neighbor features to each vertex's features.
        # out = verts_w0 + neighbor_sums
        return out

    def __repr__(self):
        d_in, d_out = self.input_dim, self.output_dim
        return "GraphConv({:d} -> {:d})".format(d_in, d_out)

    @staticmethod
    def edged_to_adj_index(edges):
        """Convert edges to the index of its adjacency.

        Args:
            edges: LongTensor of shape (E, 2) where E is the number of edges
                where each edge has the indices of the two vertices which
                form the edge.

        Returns:
            adj_index: LongTensor of shape (V, max(N)) where V is the number of
                vertices and N is the number neighbors of the vertex. The neighbor
                index.
            adj_index_: LongTensor of shape (V, max(N)) where V is the number of
                vertices and N is the number neighbors of the vertex. The center
                index.
        """
        # Flip the edge.
        flip_edges = torch.flip(edges, dims=[-1])
        # Concatenate the flipped edge and the edge.
        cat_edges = torch.cat((edges, flip_edges), dim=0)
        # Sort the concatenated edge given the first column.
        _, indices = torch.sort(cat_edges, 0)
        first_column_indices = indices[:, 0]
        sorted_edges = cat_edges[first_column_indices]
        # Get the counts of each unique index.
        _, counts = torch.unique(sorted_edges[:, 0], sorted=True, return_counts=True)
        # Split the tensor into list of tensor.
        list_index = sorted_edges[:, -1:].split(counts.cpu().detach().numpy().tolist())
        list_index_ = sorted_edges[:, :-1].split(counts.cpu().detach().numpy().tolist())
        # Convert the list of tensor to the padded tensor
        adj_index = list_to_padded(list_index, pad_value=len(list_index)).squeeze()
        adj_index_ = list_to_padded(list_index_, pad_value=len(list_index_)).squeeze()
        return adj_index, adj_index_


def get_random_cuda_device() -> str:
    """
    Function to get a random GPU device from the
    available devices. This is useful for testing
    that custom cuda kernels can support inputs on
    any device without having to set the device explicitly.
    """
    num_devices = torch.cuda.device_count()
    device_id = (
        torch.randint(high=num_devices, size=(1,)).item() if num_devices > 1 else 0
    )
    return "cuda:%d" % device_id


if __name__ == '__main__':

    dtype = torch.float32
    device = get_random_cuda_device()
    verts = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=dtype, device=device
    )
    lrfs = torch.tensor(
        [
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        ], dtype=dtype, device=device
    )
    # edges = torch.tensor([[0, 1], [0, 2]], device=device)
    edges = torch.tensor([[1, 0], [0, 2], [2, 1], [0, 3]], device=device)
    w0 = torch.tensor([[1, 1, 1]], dtype=dtype, device=device)
    # w1 = torch.tensor([[-1, -1, -1]], dtype=dtype, device=device)

    expected_y = torch.tensor(
        [
            [1 + 2 + 3 - 4 - 5 - 6 - 7 - 8 - 9],
            [4 + 5 + 6 - 1 - 2 - 3],
            [7 + 8 + 9 - 1 - 2 - 3],
        ],
        dtype=dtype,
        device=device,
    )

    conv = LRFGraphConv(3, 1).to(device)
    conv.w0.weight.data.copy_(w0)
    conv.w0.bias.data.zero_()
    # conv.w1.weight.data.copy_(w1)
    # conv.w1.bias.data.zero_()

    y = conv(verts, edges, lrfs)
    print(y, expected_y)

