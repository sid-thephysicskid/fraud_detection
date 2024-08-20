import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # binary feature for high-risk entry modes
        X['high_risk_entry_mode'] = X['posEntryMode'].isin([0, 1, 2, 81, 90]).astype(int)
        # bin transaction amounts
        X['amount_bucket'] = pd.cut(X['transactionAmount'],
                                    bins=[0, 50, 100, 150, 200, 500, 1000, np.inf],
                                    labels=[0, 1, 2, 3, 4, 5, 6])
        # Combine high-risk entry mode with high transaction amount
        X['high_risk_high_amount'] = ((X['high_risk_entry_mode'] == 1) &
                                      (X['transactionAmount'] > 100)).astype(int)

        # Extract time-based features
        X['hour'] = X['transactionTime'].dt.hour
        X['day'] = X['transactionTime'].dt.day
        X['month'] = X['transactionTime'].dt.month
        X['dayofweek'] = X['transactionTime'].dt.dayofweek
        X['is_weekend'] = X['dayofweek'].isin([5, 6]).astype(int)
        X['is_night'] = ((X['hour'] >= 22) | (X['hour'] < 5)).astype(int)
        return X
