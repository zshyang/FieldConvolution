import torch
from torch.nn import Parameter
# from torch_scatter import scatter_add, scatter_max
# from torch_geometric.utils import softmax
import os
import math
from pytorch3d.structures import list_to_padded, packed_to_list


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


# from ...utils.num_nodes import maybe_num_nodes


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class TopKPooling(torch.nn.Module):
    """TopK pooling operator from Graph U-Nets.
    """
    def __init__(
            self,
            in_channels: int,
            k: int,
            nonlinearity: torch.nn.functional = torch.tanh,
    ):
        """The initialization function.

        Args:
            in_channels: Size of each input sample.
            k: The number of left vertices.
            nonlinearity: The nonlinearity to use.
        """
        super(TopKPooling, self).__init__()

        self.in_channels = in_channels
        self.k = k
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def ref_forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """The forward function.

        Args:
            x:
            edge_index:
            edge_attr:
            batch:
            attn:

        Returns:

        """

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def forward(
            self, fea: torch.Tensor, num_vert_mesh: list,
            edge_index: torch.Tensor, vert_attr: dict
    ):
        """Forward function.

        Args:
            fea: The feature to compute pooling score.
            num_vert_mesh: Number of vertices per mesh.
            edge_index: The edges index.
            vert_attr: The features associated with vertices.

        Returns:
            fea: The pooled features.
            num_vert_mesh: The updated number of vertices per mesh.
            new_edge_index: New edge index.
            vert_attr: The pooled vertices features.
        """
        # Project the feature.
        score = (fea * self.weight).sum(dim=-1)  # (N)
        # (sum(Ni), 1)
        score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1)).view([-1, 1])
        num_nodes = score.size(0)

        # Sort the score each batch.
        # (B, max(Ni), 1)
        # TODO: The padding value is depending on the activate function.
        padded_score = list_to_padded(packed_to_list(score, num_vert_mesh), pad_value=-1.0)
        padded_score = padded_score.squeeze(-1)  # (B, max(Ni))
        batch_size, max_num_nodes = padded_score.size(0), padded_score.size(1)
        _, perm = padded_score.sort(dim=-1, descending=True)
        # A list contain tensor with size (k)
        perm = [perm[i, :self.k] + i * max_num_nodes for i in range(batch_size)]
        perm = torch.cat(perm, dim=-1)  # (k * B)

        # Update the feature.
        fea = fea[perm] * score[perm].view(-1, 1)
        num_vert_mesh = [self.k for _ in range(batch_size)]

        # Update the vertices attribute.
        vert_attr = {key: value[perm] for key, value in vert_attr.items()}

        # Rebuild the edge.
        mask = perm.new_full((num_nodes,), -1)  # (sum(Ni))
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)  # (sum(Ni))
        mask[perm] = i  # (sum(Ni))
        row, col = edge_index[:, 0], edge_index[:, 1]  # Ne, Ne
        row, col = mask[row], mask[col]  # Ne, Ne
        mask = (row >= 0) & (col >= 0)  # Ne
        row, col = row[mask], col[mask]  # Ne'
        new_edge_index = torch.stack([row, col], dim=1)  # (Ne', 2)

        return fea, num_vert_mesh, new_edge_index, vert_attr

    def __repr__(self):
        return '{}({}, {}={})'.format(
            self.__class__.__name__, self.in_channels,
            "k", self.k,
        )


def get_random_cuda_device() -> str:
    """Function to get a random GPU device from the
    available devices. This is useful for testing
    that custom cuda kernels can support inputs on
    any device without having to set the device explicitly.
    """
    num_devices = torch.cuda.device_count()
    device_id = (
        torch.randint(high=num_devices, size=(1,)).item() if num_devices > 1 else 0
    )
    return "cuda:%d" % device_id


def test0():
    dtype = torch.float32
    device = get_random_cuda_device()
    verts = torch.tensor(
        [
            [1, 2, 3], [-4, -5, 6], [-7, 8, -9], [-10, -11, 12],
            [13, -14, 15], [16, -4, -18], [19, -3, -21],
        ], dtype=dtype, device=device
    ) * 0.1
    lrfs = torch.tensor(
        [
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        ], dtype=dtype, device=device
    )
    vert_attr = {"lrf": lrfs}
    edges = torch.tensor(
        [[1, 0], [1, 2], [1, 3], [4, 5], [5, 6], [4, 6]],
        device=device
    )  # (7, 2)
    w = torch.tensor([[0.0001, 0.0001, 0.0001]], dtype=dtype, device=device)  # (1, 3)
    num_vert_mesh = [4, 3]  # Number of vertices per mesh.

    pooling_layer = TopKPooling(
        in_channels=3,
        k=2,
    ).to(device)
    pooling_layer.weight.data.copy_(w)

    verts_out, num_vert_mesh, edges, vert_attr = pooling_layer(
        verts, num_vert_mesh, edges, vert_attr
    )
    print(verts_out)
    print(num_vert_mesh)
    print(edges)
    print(vert_attr)


def test1():
    """The test function for pooling on the brain mesh.
    """
    dtype = torch.float32
    device = get_random_cuda_device()
    mesh_fn = "/home/george/PycharmProjects/shapecad/data/surf/AD_pos/002_S_0729_I291876/lh.obj"
    import trimesh
    from pytorch3d.structures import Meshes
    mesh = trimesh.load(mesh_fn, force="mesh")
    vert_list = [torch.tensor(mesh.vertices, dtype=torch.float32)]
    face_list = [torch.tensor(mesh.faces)]
    mesh = Meshes(verts=vert_list, faces=face_list).to(device)
    lrf_name = "/home/george/PycharmProjects/shapecad/data/lrf/AD_pos/002_S_0729_I291876/lh.json"
    import json
    import numpy as np
    with open(lrf_name, "r") as file:
        lrf = json.load(file)
    lrf = np.array(lrf, dtype=np.float32)
    lrf = torch.cat([torch.from_numpy(lrf)], dim=0).to(device)
    print(lrf.shape)

    # Load the mesh go through lrf conv layer.
    from torch.nn import functional as F
    import sys
    sys.path.insert(0, "/home/george/PycharmProjects/shapecad/code")
    from models.layers.lrfgconv import LRFGraphConv
    conv1 = LRFGraphConv(input_dim=3, output_dim=32).to(device)
    out1 = F.relu(
        conv1(
            verts=mesh.verts_packed(),
            edges=mesh.edges_packed(),
            lrf=lrf,
        )
    )  # (V, 32)
    print(out1.shape)

    # Do the pooling on the mesh.
    num_vert_mesh = [110150]  # Number of vertices per mesh.

    pooling_layer = TopKPooling(
        in_channels=32,
        k=128*128,
    ).to(device)
    verts_out, num_vert_mesh, edges, vert_attr = pooling_layer(
        fea=out1, num_vert_mesh=num_vert_mesh,
        edge_index=mesh.edges_packed(), vert_attr={"vert": mesh.verts_packed()}
    )
    print(verts_out.shape)
    print(num_vert_mesh)
    print(edges.shape)
    print(vert_attr)

    # Save the pooled mesh.
    edges = edges.cpu().data.numpy().tolist()
    verts = vert_attr["vert"].cpu().data.numpy()
    # print(edges.shape)
    # faces = []
    # for edge in edges:
    #     edge_0 = edge[0]
    #     edge_1 = edge[1]
    #     i = -1
    #     for i in range(len(faces)):
    #         if edge_0 in faces[i]:
    #             if len(faces[i]) == 2:
    #                 # Get the left one.
    #                 if faces[i][0] == edge_0:
    #                     left = faces[i][1]
    #                 else:
    #                     left = faces[i][0]
    #                 # Insert if possible.
    #                 if left == edge_1:
    #                     break
    #                 else:
    #                     if ([left, edge_1] in edges) or ([edge_1, left] in edges):
    #                         faces[i].extend([edge_1])
    #     i = i + 1
    #     if i < 0:
    #         faces.append(edge)
    #     if i == len(faces):
    #         faces.append(edge)
    # tri_faces = []
    # for face in faces:
    #     if len(face) == 3:
    #         tri_faces.append(face)
    #
    # with open("test.obj", "w") as file:
    #     for vert in verts:
    #         file.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
    #     for face in tri_faces:
    #         file.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for edge in edges:
        ax.plot(
            [verts[edge[0], 0], verts[edge[1], 0]],
            [verts[edge[0], 1], verts[edge[1], 1]],
            zs=[verts[edge[0], 2], verts[edge[1], 2]],
        )
    plt.show()


def test2():
    edges = [
        [0, 1],
        [1, 2],
        [1, 3],
        [0, 2],
        [2, 3],
    ]
    faces = []
    for edge in edges:
        print(edge)
        edge_0 = edge[0]
        edge_1 = edge[1]
        i = -1

        for i in range(len(faces)):
            if edge_0 in faces[i]:
                if len(faces[i]) == 2:
                    # Get the left one.
                    if faces[i][0] == edge_0:
                        left = faces[i][1]
                    else:
                        left = faces[i][0]
                    # Insert if possible.
                    if left == edge_1:
                        break
                    else:
                        if ([left, edge_1] in edges) or ([edge_1, left] in edges):
                            faces[i].extend([edge_1])

        i = i + 1
        if i < 0:
            faces.append(edge)
        if i == len(faces):
            faces.append(edge)

    print(faces)

def test3():
    VecStart_x = [0, 1, 3, 5]
    VecStart_y = [2, 2, 5, 5]
    VecStart_z = [0, 1, 1, 5]
    VecEnd_x = [1, 2, -1, 6]
    VecEnd_y = [3, 1, -2, 7]
    VecEnd_z = [1, 0, 4, 9]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(4):
        ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]], zs=[VecStart_z[i], VecEnd_z[i]])
    plt.show()
    Axes3D.plot()




if __name__ == "__main__":

    test1()






