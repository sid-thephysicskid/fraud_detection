import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from feature_engineering import CustomFeatureEngineer

class FraudDetectionPipeline:
    def __init__(self):
        self.pipeline = self.create_pipeline()

    def create_pipeline(self) -> ImbPipeline:
        self.numeric_features = ['transactionAmount', 'availableCash', 'transaction_to_cash_ratio']
        self.categorical_features = ['merchantCountry', 'mcc', 'high_risk_entry_mode', 'amount_bucket']
        self.binary_features = ['is_weekend', 'is_night', 'high_risk_high_amount', 'merchantZip_missing']
        # define preprocessing steps
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features),
            ('bin', 'passthrough', self.binary_features)
        ])
        # create full pipeline
        return ImbPipeline([
            ('feature_engineer', CustomFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy=0.1, 
                            random_state=2)),
            ('classifier', XGBClassifier(scale_pos_weight=100, 
                                         objective='binary:logistic',
                                         eval_metric='auc', tree_method='hist',
                                         random_state=2, n_jobs=-1))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

