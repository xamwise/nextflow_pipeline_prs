"""
PRS-Net (Interpretable polygenic risk scores via geometric learning), reimplemented
without DGL so it runs on the same torch stack as the rest of the model zoo.

Reference: https://github.com/lihan97/PRS-Net

Why not just import the upstream module?
-----------------------------------------
Upstream pins `dgl=1.1.0.cu116` on Python 3.7 / torch 1.13. That will not build
against sm_90 (H100). More importantly, upstream calls
``dgl.batch([ggi_graph] * batch_size)`` on every forward pass, which materialises
B copies of a ~20k-node graph per step. Because the graph is *identical* for every
sample, the GIN aggregation is just a single sparse matmul against a shared
adjacency, applied to a [B, G, d] tensor. That is both exact and dramatically
cheaper.

Input contract
--------------
forward(x) where x is [B, n_genes, d_input] (or flattened [B, n_genes * d_input]),
returning logits [B, output_dim]. This matches `predictions = self.model(genotypes)`
in train_model.py, so no Trainer changes are needed.

Gene attention weights from the readout are stashed on `.last_gene_attention`
after each forward for the interpretability analyses.
"""

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def bert_init_params(module: nn.Module) -> None:
    """Upstream's initialisation scheme, kept identical for comparability."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm1d):
        # affine=False (used by pre_bn) leaves weight/bias as None.
        if module.weight is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class MLP(nn.Module):
    """
    Feed-forward stack that accepts [..., d] and applies BatchNorm1d over the
    feature axis by flattening leading dims.

    NOTE ON `bn_bug_compat`
    -----------------------
    Upstream writes ``self.batch_norm_list[i](h)`` without assigning the result,
    so every MLP-internal BatchNorm is a no-op on the forward signal (it only
    updates running statistics). Set ``bn_bug_compat=True`` to reproduce the
    published behaviour exactly; leave it False for the corrected version.
    Report which one you used -- it is a real difference in the architecture.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        n_layers: int,
        activation: nn.Module = None,
        bias: bool = False,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        out_batchnorm: bool = False,
        bn_bug_compat: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.use_batchnorm = use_batchnorm
        self.out_batchnorm = out_batchnorm
        self.bn_bug_compat = bn_bug_compat
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.linear_list = nn.ModuleList()
        dims = []
        if n_layers == 1:
            self.linear_list.append(nn.Linear(d_input, d_output, bias=bias))
            dims = [d_output]
        else:
            self.linear_list.append(nn.Linear(d_input, d_hidden, bias=bias))
            dims.append(d_hidden)
            for _ in range(n_layers - 2):
                self.linear_list.append(nn.Linear(d_hidden, d_hidden, bias=bias))
                dims.append(d_hidden)
            self.linear_list.append(nn.Linear(d_hidden, d_output, bias=bias))
            dims.append(d_output)

        if use_batchnorm:
            # Build ONLY the BatchNorms that forward() actually applies: the
            # hidden layers, plus the output layer iff out_batchnorm. Building
            # the rest leaves dead parameters, which DDP rejects unless
            # find_unused_parameters=True. (Upstream sizes every BN to d_hidden,
            # which would crash on the output layer if the result were used.)
            n_bn = self.n_layers if out_batchnorm else self.n_layers - 1
            self.batch_norm_list = nn.ModuleList(
                [nn.BatchNorm1d(dims[i]) for i in range(n_bn)]
            )

    def _bn(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        bn = self.batch_norm_list[idx]
        shape = h.shape
        out = bn(h.reshape(-1, shape[-1])).reshape(shape)
        return h if self.bn_bug_compat else out

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers - 1):
            h = self.linear_list[i](h)
            if self.use_batchnorm:
                h = self._bn(i, h)
            h = self.dropout(self.activation(h))
        h = self.linear_list[-1](h)
        if self.use_batchnorm and self.out_batchnorm:
            h = self._bn(len(self.batch_norm_list) - 1, h)
        if self.out_batchnorm:
            h = self.dropout(self.activation(h))
        return h


class SharedGraphGINConv(nn.Module):
    """
    GIN convolution against a single adjacency shared by every sample in the batch.

    Equivalent to dgl.nn.GINConv(aggregator_type='sum', learn_eps=False) applied to
    dgl.batch([g] * B), but computed as one sparse matmul.
    """

    def __init__(self, mlp: nn.Module, learn_eps: bool = False):
        super().__init__()
        self.mlp = mlp
        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

    def forward(self, adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # h: [B, G, d]
        B, G, d = h.shape
        flat = h.permute(1, 0, 2).reshape(G, B * d)      # [G, B*d]
        agg = torch.sparse.mm(adj, flat)                 # [G, B*d]
        agg = agg.reshape(G, B, d).permute(1, 0, 2)      # [B, G, d]
        return self.mlp((1.0 + self.eps) * h + agg)


class AttentiveReadout(nn.Module):
    """Sigmoid-gated weighted sum over the gene axis."""

    def __init__(self, in_feats: int):
        super().__init__()
        self.key_layer = nn.Linear(in_feats, in_feats)
        self.query_layer = nn.Sequential(nn.Linear(in_feats, 1, bias=False), nn.Sigmoid())
        self.value_layer = nn.Linear(in_feats, in_feats)

    def forward(self, h: torch.Tensor, ancestries: Optional[torch.Tensor] = None):
        w = self.query_layer(self.key_layer(h))          # [B, G, 1]
        v = self.value_layer(h)                          # [B, G, d]
        return (v * w).sum(dim=1), w.squeeze(-1)


class AttentiveReadoutMOE(nn.Module):
    """
    Two-headed readout: one phenotype-generic query, one ancestry-conditioned query.

    This is the component that matters for cross-ancestry work -- the ancestry
    embedding lets the pooling reweight genes per ancestry group. Upstream's
    einsum "BD,BND->BN" against a [1, D] phenotype query only works when B == 1;
    the query is explicitly expanded here.
    """

    def __init__(self, in_feats: int, n_ancestries: int = 4):
        super().__init__()
        self.in_feats = in_feats
        self.ph_key_layer = nn.Linear(in_feats, in_feats)
        self.ph_value_layer = nn.Linear(in_feats, in_feats)
        self.ph_query_layer = nn.Embedding(1, in_feats)

        self.ancestry_key_layer = nn.Linear(in_feats, in_feats)
        self.ancestry_value_layer = nn.Linear(in_feats, in_feats)
        self.ancestry_query_layer = nn.Embedding(n_ancestries, in_feats)

    def forward(self, h: torch.Tensor, ancestries: torch.Tensor):
        B = h.shape[0]
        ph_q = self.ph_query_layer.weight.expand(B, -1)                  # [B, d]
        anc_q = self.ancestry_query_layer(ancestries.reshape(-1).long())  # [B, d]

        ph_w = torch.sigmoid(torch.einsum("bd,bnd->bn", ph_q, self.ph_key_layer(h)))
        anc_w = torch.sigmoid(torch.einsum("bd,bnd->bn", anc_q, self.ancestry_key_layer(h)))

        ph_h = (self.ph_value_layer(h) * ph_w.unsqueeze(-1)).sum(dim=1)
        anc_h = (self.ancestry_value_layer(h) * anc_w.unsqueeze(-1)).sum(dim=1)
        return ph_h + anc_h, torch.stack([ph_w, anc_w], dim=-1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class PRSNet(nn.Module):
    """
    Gene encoder -> gene embedding -> GIN over the GGI graph -> attentive readout
    -> predictor.
    """

    def __init__(
        self,
        n_genes: int,
        edge_index: np.ndarray,
        d_input: int = 11,
        d_hidden: int = 64,
        output_dim: int = 1,
        n_gene_encode_layer: int = 1,
        n_layers: int = 1,
        n_predictor_layer: int = 2,
        mlp_hidden_ratio: int = 1,
        pre_norm: bool = False,
        dropout: float = 0.0,
        multiple_ancestries: bool = False,
        n_ancestries: int = 4,
        add_self_loops: bool = False,
        symmetrize: str = "auto",
        preserve_multiplicity: bool = True,
        bn_bug_compat: bool = False,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.pre_norm = pre_norm
        self.multiple_ancestries = multiple_ancestries
        self.activation = nn.GELU()

        # ---- adjacency (stored as buffers so DDP broadcasts it correctly) ----
        ei = np.asarray(edge_index, dtype=np.int64)
        if ei.shape[0] != 2:
            ei = ei.T

        # PARALLEL EDGES ARE MEANINGFUL.
        # The upstream GGI graph is a multigraph: 250,236 edge entries over 134,992
        # unique (src, dst) pairs, with multiplicity up to 5. dgl GINConv with
        # aggregator_type='sum' sums over EVERY edge, so multiplicity acts as an
        # integer edge weight. Deduplicating changes the per-node aggregation
        # magnitude by up to 4x for ~42% of nodes -- a real behavioural difference,
        # not a cosmetic one. Keeping duplicates and letting coalesce() sum them
        # reproduces dgl exactly.
        if symmetrize == "auto":
            fwd = set(zip(ei[0].tolist(), ei[1].tolist()))
            do_sym = not all((d, s_) in fwd for s_, d in list(fwd)[:20000])
        else:
            do_sym = bool(symmetrize)
        if do_sym:
            ei = np.concatenate([ei, ei[::-1]], axis=1)
        if add_self_loops:
            loops = np.arange(n_genes, dtype=np.int64)
            ei = np.concatenate([ei, np.stack([loops, loops])], axis=1)
        if not preserve_multiplicity:
            ei = np.unique(ei, axis=1)
        if ei.size and ei.max() >= n_genes:
            raise ValueError(
                f"edge_index references node {ei.max()} but n_genes={n_genes}. "
                "The GGI graph and the gene feature ordering are out of sync."
            )
        self.register_buffer("edge_index", torch.from_numpy(ei), persistent=True)
        self._adj_cache = None

        # ---- gene encoder ----
        if pre_norm:
            self.pre_bn = nn.BatchNorm1d(d_input, affine=False)
        self.gene_encoder = MLP(
            d_input=d_input, d_hidden=d_hidden, d_output=d_hidden,
            n_layers=n_gene_encode_layer, activation=self.activation, bias=True,
            use_batchnorm=True, out_batchnorm=True, bn_bug_compat=bn_bug_compat,
        )
        self.gene_embeddings = nn.Embedding(n_genes, d_hidden)

        # ---- GIN stack ----
        self.gnn_layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for _ in range(n_layers):
            mlp = MLP(
                d_hidden, d_hidden * mlp_hidden_ratio, d_hidden,
                n_layers=n_gene_encode_layer, activation=self.activation, bias=False,
                use_batchnorm=True, out_batchnorm=False, bn_bug_compat=bn_bug_compat,
            )
            self.gnn_layer_list.append(SharedGraphGINConv(mlp, learn_eps=False))
            self.batch_norm_list.append(nn.BatchNorm1d(d_hidden))

        # ---- readout + predictor ----
        if multiple_ancestries:
            self.readout = AttentiveReadoutMOE(d_hidden, n_ancestries=n_ancestries)
        else:
            self.readout = AttentiveReadout(d_hidden)

        self.predictor = MLP(
            d_input=d_hidden, d_hidden=d_hidden, d_output=output_dim,
            n_layers=n_predictor_layer, dropout=dropout, activation=self.activation,
            bias=True, use_batchnorm=True, out_batchnorm=False, bn_bug_compat=bn_bug_compat,
        )

        self.apply(bert_init_params)

    def _adjacency(self) -> torch.Tensor:
        dev = self.edge_index.device
        if self._adj_cache is None or self._adj_cache.device != dev:
            vals = torch.ones(self.edge_index.shape[1], device=dev)
            self._adj_cache = torch.sparse_coo_tensor(
                self.edge_index, vals, (self.n_genes, self.n_genes)
            ).coalesce()
        return self._adj_cache

    def forward(self, x: torch.Tensor, ancestries: Optional[torch.Tensor] = None):
        if x.dim() == 2:
            x = x.reshape(x.shape[0], -1, self.d_input)
        B, G, _ = x.shape

        if self.pre_norm:
            x = self.pre_bn(x.reshape(-1, self.d_input)).reshape(B, G, self.d_input)

        h = self.gene_encoder(x) + self.gene_embeddings.weight  # [B, G, d]

        adj = self._adjacency()
        for i in range(self.n_layers):
            h = self.gnn_layer_list[i](adj, h)
            h = self.batch_norm_list[i](h.reshape(-1, self.d_hidden)).reshape(B, G, self.d_hidden)
            h = F.gelu(h)

        g_h, weights = self.readout(h, ancestries)
        return self.predictor(g_h), weights


class PRSNetWrapper(nn.Module):
    """
    Adapter that gives PRSNet the single-tensor-in / single-tensor-out signature
    the rest of the pipeline expects.

    Attention weights are kept on `.last_gene_attention` rather than returned, so
    Trainer.train_epoch / validate work unmodified.
    """

    def __init__(self, graph_file: str, input_dim: int, n_channels: int = 11,
                 output_dim: int = 1, ancestry_file: Optional[str] = None, **kwargs):
        """kwargs are forwarded to PRSNet; see its signature."""
        super().__init__()
        blob = np.load(graph_file, allow_pickle=False)
        edge_index = blob["edge_index"]
        n_graph_nodes = int(blob["n_nodes"])

        # The caveat travels with the artifact: a graph built with
        # --truncate_to_genes records alignment_verified=False, so a plumbing
        # graph cannot quietly become a result.
        if "alignment_verified" in blob and not bool(blob["alignment_verified"]):
            warnings.warn(
                f"{graph_file} was built WITHOUT verified gene/node alignment "
                "(alignment_verified=False). Gene positions in the feature matrix "
                "may not correspond to graph nodes. Fine for plumbing runs; "
                "results and any attention-based interpretation from this graph "
                "are not trustworthy.",
                RuntimeWarning, stacklevel=2,
            )

        if n_graph_nodes != input_dim:
            raise ValueError(
                f"GGI graph has {n_graph_nodes} nodes but the feature tensor has "
                f"{input_dim} genes. Node order and gene order MUST correspond "
                f"element-wise -- see convert_ggi_graph.py."
            )

        self.model = PRSNet(
            n_genes=input_dim,
            edge_index=edge_index,
            d_input=n_channels,
            output_dim=output_dim,
            **kwargs,
        )
        self.last_gene_attention = None

        # Optional per-sample ancestry codes, aligned to the HDF5 sample axis.
        if ancestry_file is not None:
            codes = np.load(ancestry_file)["ancestry"].astype(np.int64)
            self.register_buffer("ancestry_codes", torch.from_numpy(codes), persistent=False)
        else:
            self.ancestry_codes = None

    def forward(self, x: torch.Tensor, sample_idx: Optional[torch.Tensor] = None):
        anc = None
        if self.ancestry_codes is not None and sample_idx is not None:
            anc = self.ancestry_codes[sample_idx.to(self.ancestry_codes.device)]
        preds, weights = self.model(x, ancestries=anc)
        self.last_gene_attention = weights.detach()
        return preds