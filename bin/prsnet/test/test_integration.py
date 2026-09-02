#!/usr/bin/env python
"""
PRS-Net integration test on synthetic data.

Builds a tiny (N, G, 11) HDF5 in the exact shape PRSNET_ASSEMBLE produces, then
pushes it through YOUR real data_splitter.py and genotype_dataset.py and into
PRSNet -- forward, loss, backward. No UK Biobank access, no PLINK, no GPU. Runs in
a few seconds.

This is the test that catches contract breakage: if get_input_dim() stops meaning
"n_genes" or the model's forward signature drifts, this fails loudly instead of
surfacing as a bad AUROC three days into a real run.

Usage:
    python tests/prsnet/test_integration.py --bin_dir bin --models_dir models

Exit code 0 = all passed.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PASS, FAIL = "  [PASS]", "  [FAIL]"
_results = []


def check(name, fn):
    try:
        detail = fn()
        print(f"{PASS} {name}" + (f" -- {detail}" if detail else ""))
        _results.append(True)
    except Exception as e:
        print(f"{FAIL} {name}\n         {type(e).__name__}: {e}")
        _results.append(False)


def build_synthetic(tmp, n_samples=240, n_genes=50, n_thr=11, seed=0):
    """Write a genotype_data.h5 / phenotypes.csv pair matching PRSNET_ASSEMBLE."""
    import h5py
    import pandas as pd

    rng = np.random.default_rng(seed)
    feats = rng.normal(0, 0.01, (n_samples, n_genes, n_thr)).astype(np.float32)
    # Some genes carry no surviving clump -> exactly-zero rows, as in real output.
    feats[:, :5, :] = 0.0
    # Plant signal in a few genes so the loss can actually move.
    signal = feats[:, 10:15, 5].sum(axis=1) * 40
    y = (signal + rng.normal(0, 0.2, n_samples) > 0).astype(np.float64)

    h5 = Path(tmp) / "genotype_data.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("genotypes", data=feats, dtype="float32",
                         compression="gzip", compression_opts=4,
                         chunks=(1, n_genes, n_thr))
        f.attrs["n_samples"] = n_samples
        f.attrs["n_snps"] = n_genes
        f.attrs["n_genes"] = n_genes
        f.attrs["n_thresholds"] = n_thr
        f.attrs["encoding"] = "gene-prs"
        f.attrs["encoded_shape"] = np.array(feats.shape, dtype=np.int64)

    pheno = Path(tmp) / "phenotypes.csv"
    pd.DataFrame({"phenotype_1": y}).to_csv(pheno, index=False)

    graph = Path(tmp) / "ggi_graph.npz"
    e = rng.integers(0, n_genes, size=(2, 300)).astype(np.int64)
    np.savez_compressed(graph, edge_index=e, n_nodes=np.int64(n_genes))

    return h5, pheno, graph, n_samples, n_genes, n_thr





def load_prsnet_wrapper(models_dir):
    """Import PRSNetWrapper under either a package or a flat layout."""
    import importlib
    names = [f"{models_dir.name}.prsnet_model", "prsnet_model"] \
        if (models_dir / "__init__.py").exists() else ["prsnet_model"]
    for name in names:
        try:
            return importlib.import_module(name).PRSNetWrapper
        except Exception:
            continue
    raise SystemExit(
        f"cannot import prsnet_model from {models_dir}. Check the file is there "
        "and --models_dir is correct."
    )


def load_create_model(models_dir, bin_dir):
    """
    Locate create_model across layouts.

    Tries the package path FIRST ("models.models"), because that is how
    train_model.py imports it in a repo with models/__init__.py. Only falls back
    to loading a file directly when there is no package, and only then is a
    relative import actually wrong.
    """
    import importlib, importlib.util

    is_package = (models_dir / "__init__.py").exists()

    for name in ("models.models", "models", models_dir.name + ".models"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "create_model"):
            return mod.create_model

    if is_package:
        raise AssertionError(
            f"{models_dir}/__init__.py exists, so this is a package, but "
            "create_model was not found via `models.models` or `models`.\n"
            "         Check that models/models.py defines create_model and that "
            "the prsnet branch (Patch 1) was added to it.\n"
            "         Keep the relative import `from .prsnet_model import ...` -- "
            "it is correct for a package."
        )

    for cand in (models_dir / "models.py", bin_dir / "models.py"):
        if not cand.exists():
            continue
        spec = importlib.util.spec_from_file_location("prs_models_under_test", cand)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ImportError as e:
            if "relative import" in str(e):
                raise AssertionError(
                    f"{cand} uses a relative import but there is no __init__.py "
                    "beside it, so it loads as a top-level module. Either add "
                    "__init__.py, or drop the leading dot."
                )
            raise
        if hasattr(mod, "create_model"):
            return mod.create_model

    raise AssertionError(
        "could not find create_model. Pass --models_dir pointing at the directory "
        "holding models.py."
    )


def use_real(tmp, args):
    """Point the suite at artifacts produced by PRSNET_ASSEMBLE."""
    import h5py
    import pandas as pd

    h5 = Path(args.h5).resolve()
    pheno = Path(args.pheno).resolve() if args.pheno else h5.parent / "phenotypes.csv"
    if not pheno.exists():
        raise SystemExit(f"phenotypes.csv not found at {pheno}; pass --pheno")

    with h5py.File(h5, "r") as f:
        N, G, C = f["genotypes"].shape

    # A phenotype column that is entirely missing or single-class will make
    # data_splitter drop every sample. Say so plainly rather than failing deep
    # inside a stratified split.
    df = pd.read_csv(pheno)
    y = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    n_valid = int(y.notna().sum())
    n_class = int(y.dropna().nunique())
    if n_valid == 0 or n_class < 2:
        msg = (f"phenotype column has {n_valid} non-missing values across "
               f"{n_class} class(es).")
        if not args.synthetic_pheno:
            raise SystemExit(
                f"\n{msg}\nPRSNET_ASSEMBLE read phenotypes from the .fam PHENO column, "
                "which is likely -9 (missing) throughout.\n"
                "Either re-run assembly with --phenotype_file, or add --synthetic_pheno "
                "to test the mechanics with a stand-in label."
            )
        print(f"  [WARN] {msg} Substituting a synthetic binary phenotype.")
        rng = np.random.default_rng(0)
        pheno = Path(tmp) / "phenotypes.csv"
        pd.DataFrame({"phenotype_1": rng.integers(0, 2, N).astype(float)}).to_csv(
            pheno, index=False)
    else:
        print(f"  phenotype: {n_valid}/{N} non-missing, {n_class} classes")

    # The real GGI graph has 19,836 nodes and will not match a per-chromosome h5.
    # Generate one that does; this test verifies plumbing, not gene wiring.
    if args.graph:
        graph = Path(args.graph).resolve()
        n_nodes = int(np.load(graph)["n_nodes"])
        if n_nodes != G:
            raise SystemExit(
                f"--graph has {n_nodes} nodes but the h5 has {G} genes. "
                "For a per-chromosome smoke h5, omit --graph."
            )
    else:
        graph = Path(tmp) / "ggi_graph.npz"
        rng = np.random.default_rng(0)
        np.savez_compressed(graph,
                            edge_index=rng.integers(0, G, (2, G * 6)).astype(np.int64),
                            n_nodes=np.int64(G))
        print(f"  graph: generated random {G}-node stand-in (plumbing check only)")

    return h5, pheno, graph, N, G, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_dir", default="bin",
                    help="dir holding data_splitter.py and genotype_dataset.py")
    ap.add_argument("--models_dir", default="models",
                    help="dir holding prsnet_model.py")
    ap.add_argument("--h5", default=None,
                    help="Real genotype_data.h5 from PRSNET_ASSEMBLE. Omit to run "
                         "on synthetic data.")
    ap.add_argument("--pheno", default=None,
                    help="phenotypes.csv beside the h5 (defaults to that path)")
    ap.add_argument("--graph", default=None,
                    help="ggi_graph.npz. Omit and a random graph matching the h5's "
                         "gene count is generated -- fine here, since this test "
                         "checks plumbing, not biology.")
    ap.add_argument("--synthetic_pheno", action="store_true",
                    help="Replace an unusable phenotype column with a synthetic "
                         "binary one, to test the mechanics regardless.")
    args = ap.parse_args()

    bin_dir = Path(args.bin_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    sys.path.insert(0, str(bin_dir))
    sys.path.insert(0, str(models_dir.parent))
    if not (models_dir / "__init__.py").exists():
        # Flat layout only. For a package, adding models_dir would shadow the
        # package with models/models.py.
        sys.path.insert(0, str(models_dir))

    tmp = tempfile.mkdtemp(prefix="prsnet_test_")
    print(f"\nPRS-Net integration test\nworkdir: {tmp}\n" + "-" * 62)

    if args.h5:
        h5, pheno, graph, N, G, C = use_real(tmp, args)
        print(f"REAL artifacts: {h5}")
        print(f"  N={N}, n_genes={G}, n_thresholds={C}\n")
    else:
        h5, pheno, graph, N, G, C = build_synthetic(tmp)
        print(f"synthetic data: N={N}, n_genes={G}, n_thresholds={C}\n")

    # ---------------------------------------------------------------- 1
    print("1. Artifact contract")

    def t_h5():
        import h5py
        with h5py.File(h5, "r") as f:
            s = f["genotypes"].shape
            assert s == (N, G, C), f"shape {s}"
            assert f.attrs["encoding"] == "gene-prs"
            return f"genotypes {s}, encoding={f.attrs['encoding']}"
    check("HDF5 is (N, n_genes, n_thresholds) with gene-prs encoding", t_h5)

    def t_chunk():
        import h5py
        with h5py.File(h5, "r") as f:
            ch = f["genotypes"].chunks
            assert ch[0] == 1, f"chunk spans {ch[0]} samples; random reads will thrash"
            return f"chunks={ch}"
    check("HDF5 chunk is one sample wide", t_chunk)

    # ---------------------------------------------------------------- 2
    print("\n2. data_splitter.py (unmodified)")
    splits = Path(tmp) / "splits.json"
    idx = Path(tmp) / "indices.npz"

    def t_split():
        r = subprocess.run([
            sys.executable, str(bin_dir / "data_splitter.py"),
            "--genotype_file", str(h5), "--phenotype_file", str(pheno),
            "--n_folds", "3", "--stratified",
            "--output_splits", str(splits), "--output_indices", str(idx),
        ], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-600:]
        info = json.loads(splits.read_text())
        assert info["is_binary"], "phenotype not detected as binary"
        return f"train={info['n_train']} val={info['n_val']} test={info['n_test']}"
    check("splits without modification on a 3D gene-prs tensor", t_split)

    # ---------------------------------------------------------------- 3
    print("\n3. GenotypeDataModule shape contract")
    dm = None

    def t_dm():
        nonlocal dm
        from genotype_dataset import GenotypeDataModule
        dm = GenotypeDataModule(
            h5_file=str(h5), phenotype_file=str(pheno), indices_file=str(idx),
            batch_size=min(64, max(8, N // 16)), augment_train=False,
        )
        assert dm.get_input_dim() == G, f"input_dim={dm.get_input_dim()} != n_genes {G}"
        assert dm.get_n_channels() == C, f"n_channels={dm.get_n_channels()} != {C}"
        assert dm.get_output_dim() == 1, f"output_dim={dm.get_output_dim()}"
        return f"input_dim={G} n_channels={C} output_dim=1"
    check("get_input_dim/get_n_channels/get_output_dim", t_dm)

    def t_batch():
        xb, yb = next(iter(dm.train_dataloader(fold=0)))
        assert xb.shape[1:] == (G, C), f"batch {tuple(xb.shape)}"
        assert not xb.isnan().any(), "NaNs in batch"
        return f"batch x={tuple(xb.shape)} y={tuple(yb.shape)}"
    check("train_dataloader yields [B, n_genes, n_thresholds]", t_batch)

    # ---------------------------------------------------------------- 4
    print("\n4. PRSNet model")
    import torch
    PRSNetWrapper = load_prsnet_wrapper(models_dir)

    def t_build():
        m = PRSNetWrapper(graph_file=str(graph), input_dim=G, n_channels=C,
                          output_dim=1, d_hidden=32, n_layers=2)
        n = sum(p.numel() for p in m.parameters())
        return f"{n:,} parameters"
    check("builds from ggi_graph.npz", t_build)

    def t_guard():
        try:
            PRSNetWrapper(graph_file=str(graph), input_dim=G + 7, n_channels=C)
        except ValueError as e:
            assert "out of sync" in str(e) or "node" in str(e).lower()
            return "raises on gene/node count mismatch"
        raise AssertionError("accepted a mismatched gene count -- the alignment "
                             "guard is not firing")
    check("refuses a graph whose node count != n_genes", t_guard)

    def t_fwd():
        m = PRSNetWrapper(graph_file=str(graph), input_dim=G, n_channels=C, d_hidden=32)
        m.eval()
        xb, _ = next(iter(dm.val_dataloader(fold=0)))
        with torch.no_grad():
            out = m(xb)
        assert out.shape == (xb.shape[0], 1), f"{tuple(out.shape)}"
        assert m.last_gene_attention is not None
        assert m.last_gene_attention.shape[1] == G
        return f"logits {tuple(out.shape)}, attention {tuple(m.last_gene_attention.shape)}"
    check("forward on a real dataloader batch", t_fwd)

    def t_flat():
        m = PRSNetWrapper(graph_file=str(graph), input_dim=G, n_channels=C, d_hidden=32)
        m.eval()
        x = torch.randn(8, G, C)
        with torch.no_grad():
            a, b = m(x), m(x.reshape(8, -1))
        assert torch.allclose(a, b, atol=1e-6)
        return "flattened input matches 3D input"
    check("accepts [B, n_genes*n_thresholds] equivalently", t_flat)

    def t_prenorm():
        m = PRSNetWrapper(graph_file=str(graph), input_dim=G, n_channels=C,
                          d_hidden=32, pre_norm=True, bn_bug_compat=False)
        m.train()
        out = m(torch.randn(16, G, C) * 1e-3)
        assert torch.isfinite(out).all()
        return "pre_norm=True stable on 1e-3 scale features"
    check("pre_norm handles realistic PRS magnitudes", t_prenorm)

    # ---------------------------------------------------------------- 5
    print("\n5. Training step")

    def t_train():
        torch.manual_seed(0)
        m = PRSNetWrapper(graph_file=str(graph), input_dim=G, n_channels=C, d_hidden=32)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
        lossf = torch.nn.BCEWithLogitsLoss()
        loader = dm.train_dataloader(fold=0)
        first = last = None
        for epoch in range(12):
            for xb, yb in loader:
                opt.zero_grad()
                loss = lossf(m(xb), yb.float().reshape(-1, 1))
                loss.backward()
                opt.step()
                last = loss.item()
                if first is None:
                    first = last
        assert np.isfinite(last), "loss diverged"
        assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"
        return f"BCE {first:.4f} -> {last:.4f}"
    check("loss decreases over 12 epochs", t_train)

    def t_grads():
        m = PRSNetWrapper(graph_file=str(graph), input_dim=G, n_channels=C, d_hidden=32)
        torch.nn.BCEWithLogitsLoss()(m(torch.randn(8, G, C)),
                                     torch.randint(0, 2, (8, 1)).float()).backward()
        dead = [n for n, p in m.named_parameters()
                if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
        assert not dead, f"no gradient reaching: {dead[:5]}"
        return "all trainable parameters receive gradient"
    check("gradient reaches every parameter", t_grads)

    # ---------------------------------------------------------------- 6
    print("\n6. Augmentation guard")

    def t_shuffle():
        """Passes if Patch A is applied: the guard raises, or rows are left alone."""
        from genotype_dataset import GenotypeDataset
        try:
            ds = GenotypeDataset(str(h5), str(pheno), indices=np.arange(20),
                                 augment=True,
                                 augmentation_params={"shuffle_regions": 5})
            # Augmentation is stochastic: two draws can coincide by chance, so
            # sample repeatedly and look for ANY row permutation.
            ref = ds.genotypes[ds.indices[0]] if hasattr(ds, "genotypes") else None
            draws = [ds[0][0] for _ in range(40)]
            permuted = any(not torch.allclose(draws[0], d) for d in draws[1:])
        except ValueError as e:
            return f"guard raised: {str(e)[:50]}"
        if not permuted:
            return "no permutation across 40 draws"
        raise AssertionError(
            "shuffle_regions permutes gene rows, and Patch A is NOT applied. Gene "
            "position is bound to the GGI node index and to gene_embeddings, so this "
            "silently decorrelates every gene from its own neighbourhood. Either set "
            "augment_train: false, or apply Patch A from "
            "PATCHES_genotype_dataset.md."
        )
    check("[needs Patch A] shuffle_regions cannot permute gene rows", t_shuffle)


    # ---------------------------------------------------------------- 7
    print("\n7. models.py registration (Patch 1)")

    def t_registry():
        create_model = load_create_model(models_dir, bin_dir)
        cfg = {
            "model_type": "prsnet",
            "task_type": "classification",
            "input_dim": G, "output_dim": 1, "n_channels": C,
            "prsnet": {"ggi_graph": str(graph), "d_hidden": 32, "gnn_layers": 1},
        }
        try:
            m = create_model(cfg)
        except (ValueError, KeyError) as e:
            raise AssertionError(
                f"create_model rejected model_type='prsnet' ({e}). Apply Patch 1 "
                "from PRSNET_INTEGRATION.md -- the branch must read from the nested "
                "model.prsnet block, using 'gnn_layers' rather than 'n_layers'."
            )
        out = m(torch.randn(4, G, C))
        assert out.shape == (4, 1), f"{tuple(out.shape)}"
        return "create_model dispatches prsnet and returns logits"
    check("[needs Patch 1] create_model('prsnet') builds a working model", t_registry)

    def t_no_collision():
        """n_layers: 2 in the shared model: block must not reach the GIN stack."""
        create_model = load_create_model(models_dir, bin_dir)
        cfg = {
            "model_type": "prsnet",
            "task_type": "classification",
            "input_dim": G, "output_dim": 1, "n_channels": C,
            "n_layers": 2,                       # LSTM/Transformer value, top level
            "prsnet": {"ggi_graph": str(graph), "d_hidden": 32, "gnn_layers": 1},
        }
        m = create_model(cfg)
        inner = m.model if hasattr(m, "model") else m
        assert inner.n_layers == 1, (
            f"GIN stack has {inner.n_layers} layers, not 1. The branch is reading "
            "'n_layers' from the shared top-level model block instead of "
            "'gnn_layers' from model.prsnet."
        )
        return "top-level n_layers: 2 does not leak into the GIN stack"
    check("[needs Patch 1] nested config avoids the n_layers collision", t_no_collision)

    # ---------------------------------------------------------------- summary
    n_pass, n_tot = sum(_results), len(_results)
    print("\n" + "-" * 62)
    print(f"{n_pass}/{n_tot} passed")
    if n_pass != n_tot:
        print("\nA failure here means the integration contract is broken. Fix before "
              "spending compute on real data.")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())