from __future__ import annotations

import torch
from torch import nn

import matgl
from matgl.layers import BondExpansion, EmbeddingBlock
from matgl.layers._embedding import (
    TensorEmbedding,
)


def test_embedding_block_ntypes_state_float_input():
    # Regression test: EmbeddingBlock must accept float32 state_attr when ntypes_state is set.
    # MEGNet.predict_structure() builds state_attr with dtype=float32 (matgl.float_th), so
    # the .long() cast inside forward() is required to avoid a FloatTensor crash in nn.Embedding.
    embed = EmbeddingBlock(
        degree_rbf=9,
        activation=nn.SiLU(),
        dim_node_embedding=16,
        dim_edge_embedding=16,
        include_state=True,
        ntypes_state=4,
        dim_state_embedding=8,
    )
    node_attr = torch.randint(0, 2, (4,))
    edge_attr = torch.randn(6, 9)
    state_attr = torch.tensor([1], dtype=torch.float32)  # float, as built by predict_structure

    _, _, state_feat = embed(node_attr, edge_attr, state_attr)
    assert state_feat is not None
    assert state_feat.shape == (1, 8)


def test_tensor_embedding(graph_Mo):
    _, g1, state1 = graph_Mo
    bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=True)
    g1.edge_attr = bond_expansion(g1.bond_dist)
    # without state
    tensor_embedding = TensorEmbedding(
        units=64,
        degree_rbf=3,
        activation=nn.SiLU(),
        ntypes_node=1,
        cutoff=5.0,
        dtype=matgl.float_th,
    )

    X, state_feat = tensor_embedding(g1.node_type, g1.edge_index, g1.edge_attr, g1.bond_dist, g1.bond_vec, state1)

    assert [X.shape[0], X.shape[1], X.shape[2], X.shape[3]] == [2, 64, 3, 3]
    assert state_feat is None
