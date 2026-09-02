"""
Convert the upstream GGI graph binary to a portable npz -- WITHOUT DGL.

Why not use DGL
---------------
Upstream ships the graph as a serialised DGL object. Installing DGL to read one
file means matching its compiled C++ extension to your exact torch build; on a
current stack that fails with e.g.

    FileNotFoundError: Cannot find DGL C++ graphbolt library at
    .../dgl/graphbolt/libgraphbolt_pytorch_2.8.0.dylib

The payload is plain TVM NDArrays, so it can be read directly. This parser locates
the NDArray headers by magic number and decodes the two int64 edge arrays. No DGL,
no torch, no compiled extensions -- just numpy.

Output npz
----------
  edge_index  int64 [2, E]   source row, destination row
  n_nodes     int64 scalar

Node indices are POSITIONAL. Node i means "row i of the feature matrix"; the file
carries no gene names. See the alignment section below.
"""

import argparse
import os
import struct
import sys

import numpy as np

TVM_NDARRAY_MAGIC = 0xDD5E40F096B4A13F


def read_ndarrays(path):
    """Decode every TVM NDArray in a DGL-serialised file."""
    with open(path, "rb") as fh:
        blob = fh.read()

    needle = struct.pack("<Q", TVM_NDARRAY_MAGIC)
    arrays, off = [], blob.find(needle)
    while off != -1:
        p = off + 8 + 8                                  # magic + reserved
        struct.unpack_from("<ii", blob, p); p += 8        # DLDevice
        ndim, = struct.unpack_from("<i", blob, p); p += 4
        code, bits, lanes = struct.unpack_from("<BBH", blob, p); p += 4
        shape = struct.unpack_from("<" + "q" * ndim, blob, p); p += 8 * ndim
        nbytes, = struct.unpack_from("<q", blob, p); p += 8

        dtype = {(0, 64): "<i8", (0, 32): "<i4", (1, 64): "<u8",
                 (1, 32): "<u4", (2, 32): "<f4", (2, 64): "<f8"}.get((code, bits))
        if dtype is not None and ndim == 1 and nbytes == shape[0] * (bits // 8):
            arrays.append(np.frombuffer(blob, dtype=dtype, count=shape[0], offset=p))
        off = blob.find(needle, off + 1)
    return arrays


def canonical_gene_order(gene_bed_dir):
    """Sorted by chromosome number, then gene symbol. Filesystem-independent."""
    genes = []
    for chrom in range(1, 23):
        d = os.path.join(gene_bed_dir, f"chr{chrom}")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".bed"):
                genes.append((chrom, fn[:-4]))
    if not genes:
        sys.exit(
            f"FATAL: no .bed files under {gene_bed_dir!r}. Expected "
            "<gene_bed_dir>/chr1/<GENE>.bed ... chr22/<GENE>.bed"
        )
    return genes


def main():
    ap = argparse.ArgumentParser(description="DGL-free GGI graph converter")
    ap.add_argument("--ggi_graph", required=True)
    ap.add_argument("--gene_bed_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gene_order_out", default="gene_order_from_graph.txt")
    ap.add_argument("--allow_mismatch", action="store_true",
                    help="Write anyway when node count != gene count. The feature "
                         "and node axes will NOT be aligned; see below.")
    ap.add_argument("--gene_order", default=None, metavar="FILE",
                    help="The gene list that DEFINES the feature axis (one gene per "
                         "line; extra tab-separated columns ignored). Normally "
                         "gene_order.txt from build_gene_snp_map.py. When given, the "
                         "target node count is read from this file rather than "
                         "hardcoded, and gene names are cross-checked against the "
                         "BED directory.")
    ap.add_argument("--truncate", action="store_true",
                    help="PLUMBING RUNS ONLY. Truncate the graph to the target node "
                         "count derived from --gene_order (or the BED file count if "
                         "--gene_order is absent). Equivalent to passing the number "
                         "explicitly via --truncate_to_genes, but dataset-independent.")
    ap.add_argument("--truncate_to_genes", type=int, default=None, metavar="N",
                    help="PLUMBING RUNS ONLY. Drop every edge touching a node index "
                         ">= N and set n_nodes = N, so the graph loads against an "
                         "N-gene feature matrix. Which nodes get dropped is "
                         "arbitrary -- the file has no gene names, so there is no "
                         "reason to think the surplus nodes are the last ones. The "
                         "resulting npz is marked alignment_verified=False and "
                         "PRSNetWrapper will warn on every load. Do not report "
                         "results from a graph built this way.")
    args = ap.parse_args()

    arrays = read_ndarrays(args.ggi_graph)
    edge_arrays = [a for a in arrays if a.dtype == np.int64 and len(a) > 1000]
    if len(edge_arrays) < 2:
        sys.exit(f"FATAL: expected two large int64 arrays in {args.ggi_graph}, "
                 f"found {len(edge_arrays)}. Format may differ from the version "
                 "this parser was written against.")
    src, dst = edge_arrays[0], edge_arrays[1]
    if len(src) != len(dst):
        sys.exit(f"FATAL: edge arrays differ in length ({len(src)} vs {len(dst)})")

    n_nodes = int(max(src.max(), dst.max())) + 1
    edge_index = np.stack([src, dst]).astype(np.int64)

    genes = canonical_gene_order(args.gene_bed_dir)
    gene_names = [g for _, g in genes]

    # The feature axis is authoritative. Prefer its length over the BED count, and
    # flag any disagreement -- they should be identical by construction, so a
    # mismatch means the two were built from different gene sets.
    target_n = len(gene_names)
    if args.gene_order:
        with open(args.gene_order) as fh:
            axis = [ln.split("\t")[0].strip() for ln in fh if ln.strip()]
        target_n = len(axis)
        if target_n != len(gene_names):
            print(f"WARNING: --gene_order lists {target_n} genes but the BED "
                  f"directory holds {len(gene_names)}. Using the feature axis "
                  f"({target_n}).", file=sys.stderr)
        elif axis != gene_names:
            n_diff = sum(1 for a, b in zip(axis, gene_names) if a != b)
            print(f"WARNING: --gene_order and the BED directory have the same length "
                  f"but differ in {n_diff} position(s). The feature axis and the "
                  f"canonical order are not the same ordering.", file=sys.stderr)

    # ---- structure report --------------------------------------------------
    self_loops = int((src == dst).sum())
    keep = src != dst
    und = len(set(map(tuple, np.stack(
        [np.minimum(src[keep], dst[keep]), np.maximum(src[keep], dst[keep])], 1))))
    deg = np.bincount(src[keep], minlength=n_nodes)

    print(f"graph nodes          : {n_nodes}")
    print(f"directed edge entries: {len(src)}")
    print(f"self-loop entries    : {self_loops}")
    print(f"real interactions    : {und} undirected, excluding self-loops")
    print(f"degree (no self-loop): mean {deg.mean():.2f}  median {int(np.median(deg))}  "
          f"max {deg.max()}")
    print(f"  genes with NO interaction partner: {int((deg == 0).sum())} "
          f"({100 * (deg == 0).mean():.1f}%)")
    print(f"gene BED files       : {len(gene_names)}")
    print(f"feature-axis genes   : {target_n}"
          + ("  (from --gene_order)" if args.gene_order else "  (from BED count)"))

    with open(args.gene_order_out, "w") as fh:
        for chrom, gene in genes:
            fh.write(f"{gene}\tchr{chrom}\n")

    if n_nodes != target_n:
        msg = f"""
NODE/GENE COUNT MISMATCH: {n_nodes} graph nodes vs {target_n} feature-axis genes
(delta {n_nodes - target_n}).

Node indices in this file are positional and it carries NO gene-name table, so
there is no way to align the feature axis with the node axis from the shipped
artifacts alone. Wrong alignment does not raise: the model trains, reports a
plausible AUROC, and every attention-based interpretability claim becomes noise.

Resolve one of two ways:
  1. Obtain the node-order file from the authors.
  2. Rebuild the graph from a named source (STRING, HumanNet) under your own gene
     order. Preferable regardless -- it makes the graph a controlled variable and
     lets you build it on only the genes that actually carry variants.
"""
        if not args.allow_mismatch:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg + "--allow_mismatch set; writing anyway.", file=sys.stderr)

    alignment_verified = (n_nodes == target_n)

    n_keep = args.truncate_to_genes
    if n_keep is None and args.truncate:
        n_keep = target_n
    if n_keep is not None and n_keep > n_nodes:
        sys.exit(
            f"FATAL: cannot truncate to {n_keep} nodes -- the graph only has "
            f"{n_nodes}. The feature axis is LARGER than the node axis, so genes "
            "would have no corresponding node. Truncation cannot fix this; rebuild "
            "the graph on your gene set."
        )

    if n_keep is not None:
        before = edge_index.shape[1]
        keep_edges = (edge_index[0] < n_keep) & (edge_index[1] < n_keep)
        edge_index = edge_index[:, keep_edges]
        dropped_nodes = max(0, n_nodes - n_keep)
        n_nodes = n_keep
        alignment_verified = False
        print(f"\n--truncate_to_genes {n_keep}: dropped {dropped_nodes} node(s) and "
              f"{before - edge_index.shape[1]} edge entries", file=sys.stderr)
        print("  This is a PLUMBING configuration. The dropped nodes were chosen by "
              "index, not by identity.", file=sys.stderr)

    np.savez_compressed(
        args.output,
        edge_index=edge_index,
        n_nodes=np.int64(n_nodes),
        alignment_verified=np.bool_(alignment_verified),
        n_gene_bed_files=np.int64(len(gene_names)),
    )
    print(f"\nWrote {args.output} and {args.gene_order_out}")
    if not alignment_verified:
        print("  NOTE: alignment_verified=False recorded in the npz.", file=sys.stderr)


if __name__ == "__main__":
    main()