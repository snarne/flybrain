"""The space of conditions a robot is tested under: named dimensions with a range and a nominal value."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Dim:
    name: str
    lo: float
    hi: float
    nominal: float
    text: str = ""


def nominal(space):
    return {d.name: d.nominal for d in space}


def to_params(space, u):
    """u in [0, 1]^n -> parameter dict."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return {d.name: float(d.lo + (d.hi - d.lo) * v) for d, v in zip(space, u)}


def to_unit(space, p):
    return np.array([(p[d.name] - d.lo) / (d.hi - d.lo) for d in space])


def severity(space, p, skip=()):
    """How far from the nominal conditions (0 = nominal, ~1 = the far end of one range), root sum of squares."""
    v = [((p[d.name] - d.nominal) / (d.hi - d.lo)) for d in space if d.name not in skip]
    return float(np.sqrt(np.sum(np.square(v))))
