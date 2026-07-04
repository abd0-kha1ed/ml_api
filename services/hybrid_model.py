import numpy as np



BASE_COLS_RIDGE = [
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "T2M_clim",
    "WS2M_clim",
    "GHI_clim",
]


class HybridModel:
    def __init__(self, lgbm, ridge, meta, scaler, ridge_cols=None):
        self.lgbm = lgbm
        self.ridge = ridge
        self.meta = meta
        self.scaler = scaler
        self.ridge_cols = ridge_cols or BASE_COLS_RIDGE

    def _ridge_cols(self, X):
        return [c for c in self.ridge_cols if c in X.columns]

    def predict(self, X):
        lgbm_pred = self.lgbm.predict(X)

        ridge_cols = self._ridge_cols(X)
        X_ridge = self.scaler.transform(X[ridge_cols])
        ridge_pred = self.ridge.predict(X_ridge)

        meta_X = np.column_stack([lgbm_pred, ridge_pred])
        return self.meta.predict(meta_X)