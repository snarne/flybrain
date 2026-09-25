"""
Download (first run only) and load the BANC brain-and-nerve-cord connectome as CSR arrays.

BANC (Bates, Phelps, Kim, Yang et al. 2026) is one adult female fly's whole central nervous system:
brain, optic lobes and ventral nerve cord, including the motor neurons that drive the legs, wings,
neck and proboscis. The public neuron-to-neuron edge list (~300 MB) is fetched from the BANC
Google Cloud bucket and packed into server/data/connectome_banc888.npz, using the neuron order in
server/data/neurons.npz (built by tools/build_banc.py).
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[1]
BASE = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/"
EDGES = "banc_888_edgelist_simple_v2.feather"
CACHE = ROOT / ".cache" / "banc"
CSR = DATA / "connectome_banc888.npz"


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url}")
    shown = [-1]

    def hook(blocks, bs, total):
        if total > 0:
            done = min(blocks * bs, total)
            step = int(40 * done / total)
            if step != shown[0]:
                shown[0] = step
                sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    tmp.rename(dest)


def build_csr(edge_file):
    import pandas as pd

    nz = np.load(DATA / "neurons.npz")
    rid = nz["root_ids"].astype(np.int64)
    sign = nz["sign"].astype(np.int32)
    order = np.argsort(rid)
    df = pd.read_feather(edge_file, columns=["pre", "post", "count"])
    pre_id = df["pre"].astype("int64").to_numpy()
    post_id = df["post"].astype("int64").to_numpy()
    cnt = df["count"].to_numpy(np.int32)
    del df

    def index_of(ids):
        k = np.searchsorted(rid, ids, sorter=order)
        k = np.clip(k, 0, len(rid) - 1)
        i = order[k]
        return np.where(rid[i] == ids, i, -1)

    pre, post = index_of(pre_id), index_of(post_id)
    ok = (pre >= 0) & (post >= 0) & (pre != post) & (cnt > 0)   # drop autapses and edges to excluded cells
    pre, post, cnt = pre[ok], post[ok], np.minimum(cnt[ok], 32000)
    # known gap junctions (giant fibre -> jump motor neuron and PSI), as excitatory synapse-equivalents
    import json
    el = np.array(json.load(open(DATA / "presets.json")).get("electrical", []), np.int64).reshape(-1, 3)
    is_el = np.concatenate([np.zeros(len(pre), bool), np.ones(len(el), bool)])
    pre = np.concatenate([pre, el[:, 0]])
    post = np.concatenate([post, el[:, 1]])
    cnt = np.concatenate([cnt, el[:, 2]]).astype(np.int32)
    o = np.argsort(pre, kind="stable")
    pre, post, cnt, is_el = pre[o], post[o], cnt[o], is_el[o]
    indptr = np.zeros(len(rid) + 1, np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)
    w = np.where(is_el, cnt, cnt * sign[pre]).astype(np.int16)   # electrical coupling is always excitatory
    np.savez(CSR, indptr=indptr, indices=post.astype(np.int32), weights=w)
    print(f"  {len(rid):,} neurons, {len(w):,} connections, {int(cnt.sum()):,} synapses")


def load():
    """Returns indptr, indices, signed synapse counts (sign from each presynaptic neuron's transmitter)."""
    DATA.mkdir(exist_ok=True)
    if not CSR.exists():
        src = CACHE / EDGES
        if not src.exists():
            _download(BASE + EDGES, src)
        print("Building connectome arrays (one time)...")
        build_csr(src)
    z = np.load(CSR)
    return z["indptr"], z["indices"], z["weights"]
