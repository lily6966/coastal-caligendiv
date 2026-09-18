#!/usr/bin/env python3
"""
LULC-aware wrapper around the trained FeatureProcessor.

The saved processor is reused as-is rather than refitted, so the scaling the
pre-trained weights expect is preserved exactly; the land-use columns are
standardized separately and appended to the end of the numeric block.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

# natural_frac is omitted: it is 1 - lulc_pressure by construction.
LULC_FEATURES = ["lulc_pressure", "developed_frac", "crop_frac",
                 "land_frac", "lulc_known"]


class LulcProcessor:
    """Delegates to the base processor, then appends standardized LULC columns."""

    def __init__(self, base, features=LULC_FEATURES):
        self.base = base
        self.features = list(features)
        self.lulc_scaler = StandardScaler()
        self._fitted = False

    def __getattr__(self, name):          # metric_encoder, inverse_target, ...
        # Guard against unpickling recursion: before __dict__ is restored, `base`
        # is absent and delegating (incl. pickle dunders) must raise, not recurse.
        if name == "base" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def _extra(self, df):
        cols = []
        for f in self.features:
            v = df[f].values.astype(np.float64) if f in df.columns else np.zeros(len(df))
            cols.append(np.nan_to_num(v, nan=0.0))
        return np.column_stack(cols)

    def fit_lulc(self, df):
        self.lulc_scaler.fit(self._extra(df))
        self._fitted = True
        return self

    def transform(self, df):
        result = self.base.transform(df)
        extra = self._extra(df)
        extra = self.lulc_scaler.transform(extra) if self._fitted else extra
        result["numeric"] = np.hstack([result["numeric"], extra]).astype(np.float32)
        return result

    def fit_transform(self, df, fit=False):
        if fit:
            self.fit_lulc(df)
        return self.transform(df)


def widen_encoder(state_dict, n_base_numeric, n_new):
    """Insert n_new zero columns into the encoder's first layer.

    The feature vector is [numeric | categorical embeds | species embed], so the
    new numeric columns go in at position n_base_numeric and everything after
    shifts right. Zero weights mean the widened model starts out computing
    exactly what the pre-trained one did.
    """
    key = "feature_encoder.mlp.0.weight"
    W = state_dict[key]
    out_dim, in_dim = W.shape
    import torch
    new_W = torch.zeros(out_dim, in_dim + n_new, dtype=W.dtype)
    new_W[:, :n_base_numeric] = W[:, :n_base_numeric]
    new_W[:, n_base_numeric + n_new:] = W[:, n_base_numeric:]
    state_dict = dict(state_dict)
    state_dict[key] = new_W
    return state_dict
