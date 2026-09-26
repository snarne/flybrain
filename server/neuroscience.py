"""
Questions about the wiring, for the agent's neuroscientist tools and the neuron panel: find cell types,
describe one (what it is, what it talks to, how active it is now), and trace the strongest path from one
cell type to another. Everything here reads the BANC connectome itself (not a control variant), with
synapse counts as the connectome has them.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


class Atlas:
    def __init__(self, world):
        self.w = world
        T = world.tables
        self.names = np.array([n or "" for n in T["type"]], dtype=object)
        self.cls_names = np.array(T["class"])
        self.sup_names = np.array(T["super_class"])
        self.nt_names = np.array(T["nt"])
        self.counts = np.bincount(world.type, minlength=len(self.names))
        self._A = None
        self._TT = None

    # ------------------------------------------------------------ graphs (built on first use)
    def neuron_graph(self):
        if self._A is None:
            w = self.w
            N = w.N
            rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(w.indptr))
            self._A = sparse.csr_matrix((w.weights["paper"].astype(np.float32), (rows, w.indices)), shape=(N, N))
        return self._A

    def type_graph(self):
        """Signed synapse counts between cell types (types x types)."""
        if self._TT is None:
            A = self.neuron_graph()
            N, K = self.w.N, len(self.names)
            P = sparse.csr_matrix((np.ones(N, np.float32), (np.arange(N), self.w.type)), shape=(N, K))
            self._TT = (P.T @ A @ P).tocsr()
        return self._TT

    # ------------------------------------------------------------ lookups
    def resolve(self, name):
        """A cell type name (exact, then case-insensitive), a neuron index or a BANC root id -> (label, indices)."""
        s = str(name).strip()
        if s.isdigit():
            if s.startswith("720575") or len(s) > 7:
                m = np.where(self.w.root_ids.astype(str) == s)[0]
            else:
                m = np.array([int(s)]) if int(s) < self.w.N else np.array([], np.int64)
            if len(m):
                i = int(m[0])
                return f"neuron {i} ({self.names[self.w.type[i]] or 'unnamed'})", m.astype(np.int64)
        idx = self.w.neurons_of_type(s)
        if len(idx):
            return s, idx
        low = s.lower()
        for k, n in enumerate(self.names):
            if n and n.lower() == low:
                return n, np.where(self.w.type == k)[0]
        return None, np.array([], np.int64)

    def find(self, query, limit=20):
        """Cell types whose name, class or super class contains the query."""
        q = query.strip().lower()
        if not q:
            return []
        hits = []
        cls_of = np.bincount(self.w.type, weights=None, minlength=len(self.names))
        for k, n in enumerate(self.names):
            if n and q in n.lower():
                hits.append(k)
        if len(hits) < limit:        # also by class
            for field, table in ((self.w.cls, self.cls_names), (self.w.sup, self.sup_names)):
                match = [j for j, c in enumerate(table) if c and q in c.lower()]
                if match:
                    ks = np.unique(self.w.type[np.isin(field, match)])
                    hits += [int(k) for k in ks if self.names[k] and k not in hits]
        hits = sorted(set(hits), key=lambda k: (not self.names[k].lower().startswith(q), -cls_of[k]))
        return [self.summary(np.where(self.w.type == k)[0], self.names[k]) for k in hits[:limit]]

    def summary(self, idx, label):
        w = self.w
        maj = lambda arr, table: table[np.bincount(arr[idx]).argmax()] if len(idx) else ""
        reg = np.bincount(w.region[idx], minlength=w.n_regions)
        top = [w.tables["regions"][int(r)]["label"] for r in np.argsort(-reg)[:3] if reg[r] > 0]
        sides = {str(s) or "midline": int(c) for s, c in zip(*np.unique(np.array(w.tables["side"])[w.side[idx]], return_counts=True))}
        return {"type": label, "n": int(len(idx)), "class": maj(w.cls, self.cls_names),
                "super_class": maj(w.sup, self.sup_names), "transmitter": maj(w.nt, self.nt_names),
                "regions": top, "sides": sides,
                "hz_now": round(float(w.rate[idx].mean()), 1), "hz_max": round(float(w.rate[idx].max()), 1)}

    def partners(self, idx, direction="out", limit=8):
        """Strongest partner cell types of these neurons: synapses, sign, and share of the total."""
        A = self.neuron_graph()
        M = A[idx] if direction == "out" else A[:, idx].T.tocsr()
        coo = M.tocoo()
        other = self.w.type[coo.col]
        pos = np.bincount(other, weights=np.maximum(coo.data, 0), minlength=len(self.names))
        neg = np.bincount(other, weights=np.minimum(coo.data, 0), minlength=len(self.names))
        tot = np.abs(coo.data).sum() or 1.0
        mag = pos - neg
        out = []
        for k in np.argsort(-mag)[:limit]:
            if mag[k] <= 0:
                break
            out.append({"type": self.names[k] or "(unnamed)", "synapses": int(mag[k]),
                        "sign": "excites" if pos[k] >= -neg[k] else "inhibits", "share": round(float(mag[k] / tot), 3),
                        "hz_now": round(float(self.w.rate[self.w.type == k].mean()), 1)})
        return out

    def roles(self, idx):
        """What the app itself uses these neurons for (behaviour readouts, senses, muscles)."""
        s = set(int(i) for i in idx)
        out = []
        for r in self.w.readouts:
            if s & set(r["neurons"]):
                out.append(f"behaviour readout: {r['label']}")
        for k, st in self.w.stim.items():
            if s & set(st["neurons"]):
                out.append(f"sense: {st['label']}")
        for p in getattr(self.w.motor, "pools", []):
            if s & set(p["neurons"]):
                out.append(f"motor pool: {p['label']}")
        for g in self.w.ABLATIONS:
            if g in self.w.ablated and s & set(int(i) for i in self.w.ablated[g]):
                out.append(f"currently silenced ({g})")
        return out[:8]

    def info(self, name):
        label, idx = self.resolve(name)
        if label is None:
            return None
        d = self.summary(idx, label)
        d["inputs"] = self.partners(idx, "in")
        d["outputs"] = self.partners(idx, "out")
        d["roles"] = self.roles(idx)
        return d

    def path(self, src, dst, max_hops=6):
        """The strongest chain of cell types from src to dst: each step weighted by the share of the
        next type's input that comes from the previous one (so a path through a type's main input
        beats one through a minor one). Unnamed neurons are skipped."""
        ls, a = self.resolve(src)
        ld, b = self.resolve(dst)
        if ls is None or ld is None:
            return {"error": f"unknown cell type: {src if ls is None else dst}"}
        T = self.type_graph()
        K = T.shape[0]
        mag = abs(T)
        inflow = np.asarray(mag.sum(0)).ravel()
        coo = mag.tocoo()
        keep = (coo.row != coo.col) & (self.names[coo.row] != "") & (self.names[coo.col] != "")
        frac = coo.data[keep] / np.maximum(inflow[coo.col[keep]], 1.0)
        cost = sparse.csr_matrix((-np.log(np.clip(frac, 1e-9, 1.0)) + 1e-6, (coo.row[keep], coo.col[keep])), shape=(K, K))
        s_types = np.unique(self.w.type[a])
        d_types = set(np.unique(self.w.type[b]).tolist())
        dist, pred = dijkstra(cost, indices=s_types, return_predecessors=True, min_only=True)[:2]
        best = min(d_types, key=lambda k: dist[k])
        if not np.isfinite(dist[best]):
            return {"from": ls, "to": ld, "path": [], "note": "no path through named cell types"}
        chain = [best]
        while chain[-1] not in s_types and pred[chain[-1]] >= 0 and len(chain) < 40:
            chain.append(int(pred[chain[-1]]))
        chain.reverse()
        steps = []
        for u, v in zip(chain, chain[1:]):
            syn = float(T[u, v])
            steps.append({"from": self.names[u], "to": self.names[v], "synapses": int(abs(syn)),
                          "sign": "excites" if syn > 0 else "inhibits",
                          "share_of_input": round(float(abs(syn) / max(inflow[v], 1.0)), 3)})
        sign = "excitatory" if sum(s["sign"] == "inhibits" for s in steps) % 2 == 0 else "inhibitory (net)"
        return {"from": ls, "to": ld, "hops": len(steps), "net": sign, "path": steps,
                "note": "" if len(steps) <= max_hops else f"longer than {max_hops} steps: probably an indirect link"}


def activity_groups(world, per_row=12):
    """Neuron groups for the activity view's raster and rate plots, from the senses through the
    brain's learning and command neurons to the motor neurons. Each: key, label, stage, all neurons
    (for the rate) and a fixed sample of them (rows of the raster)."""
    S = world.presets.get("senses", {})
    cls = np.array(world.tables["class"])[world.cls]
    typ = lambda t: world.neurons_of_type(t).tolist()
    side = np.array(world.tables["side"])[world.side]
    by_side = lambda idx, s: [int(i) for i in idx if side[i] == s]
    mb = getattr(world, "mb", None)
    readout = {r["key"]: r["neurons"] for r in world.readouts}
    pools = getattr(world.motor, "pools", [])
    legs_mn = sorted({n for p in pools if p.get("part", "").endswith("_leg") for n in p["neurons"]})
    wing_mn = sorted({n for p in pools if p.get("part") == "wing" for n in p["neurons"]})
    rows = [
        ("smell", "Fruit odour receptors", "sense", S.get("odour", {}).get("fruit", {}).get("left", []) + S.get("odour", {}).get("fruit", {}).get("right", [])),
        ("loom", "Looming detectors (LPLC2, LC4)", "sense", S.get("loom", {}).get("left", []) + S.get("loom", {}).get("right", [])),
        ("target", "Small-object detectors (LC10a)", "sense", S.get("target", {}).get("left", []) + S.get("target", {}).get("right", [])),
        ("taste", "Sugar taste (labellum)", "sense", world.stim["sugar"]["neurons"] if "sugar" in world.stim else []),
        ("pn", "Antennal lobe projection neurons", "brain", np.where(cls == "antennal_lobe_projection_neuron")[0].tolist()),
        ("kc", "Kenyon cells", "brain", mb.kc.tolist() if mb is not None else []),
        ("mbon", "Mushroom body output neurons", "brain", mb.mbon.tolist() if mb is not None else []),
        ("dan", "Dopamine neurons (PAM, PPL1)", "brain", mb.dan.tolist() if mb is not None else []),
        ("gf", "Giant fibre (DNp01)", "command", typ("DNp01")),
        ("p9", "P9 (DNp09) forward", "command", typ("DNp09")),
        ("mdn", "MDN backward", "command", typ("MDN")),
        ("dna02_l", "DNa02 left turn", "command", by_side(typ("DNa02"), "left")),
        ("dna02_r", "DNa02 right turn", "command", by_side(typ("DNa02"), "right")),
        ("feed", "Feeding command", "command", readout.get("feed", [])),
        ("legs", "Leg motor neurons", "motor", legs_mn),
        ("wings", "Wing motor neurons", "motor", wing_mn),
    ]
    rng = np.random.default_rng(1)
    out = []
    for key, label, stage, idx in rows:
        idx = [int(i) for i in idx]
        if not idx:
            continue
        sample = sorted(rng.choice(idx, size=min(per_row, len(idx)), replace=False).tolist())
        out.append({"key": key, "label": label, "stage": stage, "neurons": idx, "sample": sample})
    return out
