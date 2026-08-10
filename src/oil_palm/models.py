"""SawitGuard-GNN — shared no-graph baseline model.

`layer2/models_real.py` imports `MLPBaseline` and `count_params` from here
(via a two-directories-up, `__file__`-relative `sys.path` insert — see its own
docstring) so the baseline used for the real Eg9PP panel is literally the same
class/weight-count logic wherever it's used, not a re-typed copy that could
drift.

Trimmed from the original research repo's root `models.py`: that file also
defined `STGNN` and `STGNN_SEIR`, which belonged to a synthetic SEIR-simulator
half of the project that was cut from scope before this submission (see
`docs/RESULTS.md`). Neither class was ever imported by `models_real.py` —
`layer2/models_real.py` defines its own `STGNN`/`STGNN_SID` for the real
single-relation Eg9PP panel — so they are dead code here and were dropped
rather than carried over unused.

Guards:
  * No model ever sees ground-truth S/E/I/R state as input — only spectral
    features and their neighbour aggregates (diffused features).
  * MLPBaseline uses no graph at all — the fixed field baseline.
"""
import torch.nn as nn

HIDDEN = 34


class MLPBaseline(nn.Module):
    """Per-tree, no graph. Input = node feature [pca, delta] at the query cycle."""
    def __init__(self, in_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, F_seq, D_seq):
        return self.net(F_seq[:, -1, :]).squeeze(-1)   # use last cycle only


def count_params(m):
    return sum(p.numel() for p in m.parameters())
