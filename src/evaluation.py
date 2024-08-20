import pandas as pd
import numpy as np
from model_pipeline import FraudDetectionPipeline
from sklearn.model_selection import StratifiedKFold

def time_based_train_test_split(df: pd.DataFrame, date_column: str = 'transactionTime') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into train and test sets based on time.
    Uses the last month of data as the test set.
    """
    df = df.sort_values(date_column)
    last_month = df[date_column].max().to_period('M')
    test_start_date = last_month.to_timestamp().tz_localize('UTC')

    train_df = df[df[date_column] < test_start_date]
    test_df = df[df[date_column] >= test_start_date]

    print(f"Train set date range: {train_df[date_column].min()} to {train_df[date_column].max()}")
    print(f"Test set date range: {test_df[date_column].min()} to {test_df[date_column].max()}")
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    return train_df, test_df

def fraud_capture_score(y_true: pd.Series, y_pred_proba: np.ndarray, transaction_amounts: pd.Series) -> float:
    """
    Calculates the fraud capture score based on the top 400 highest-scored transactions.
    """
    df = pd.DataFrame({
        'true_label': y_true,
        'fraud_probability': y_pred_proba,
        'transaction_value': transaction_amounts
    })
    df['fraud_score'] = df['fraud_probability'] * df['transaction_value']
    top_400 = df.nlargest(400, 'fraud_score')
    total_fraud_caught = top_400[top_400['true_label'] == 1]['transaction_value'].sum()
    total_fraud = df[df['true_label'] == 1]['transaction_value'].sum()
    print(f"Total Fraud in test data: {total_fraud:.2f}")
    print(f"Total Fraud Caught: {total_fraud_caught:.2f}")
    return total_fraud_caught / total_fraud if total_fraud > 0 else 0

def custom_fraud_threshold(y_pred_proba: np.ndarray, transaction_amounts: pd.Series, n: int = 400) -> np.ndarray:
    """
    Determines a custom threshold for fraud detection based on the top n transactions by fraud score.
    """
    df = pd.DataFrame({
        'fraud_probability': y_pred_proba,
        'transaction_value': transaction_amounts
    })
    df['fraud_score'] = df['fraud_probability'] * df['transaction_value']
    threshold = df.nlargest(n, 'fraud_score')['fraud_score'].min()
    return (df['fraud_score'] >= threshold).astype(int)

def rule_based_fraud_detection(df: pd.DataFrame, threshold: float = 150) -> pd.Series:
    """
    Implements a simple rule-based fraud flagging algorithm.
    """
    return ((df['posEntryMode'].isin([0, 1, 2, 81, 90])) &
            (df['transactionAmount'] > threshold)).astype(int)

def evaluate_model(y_true: pd.Series, y_pred_proba: np.ndarray, transaction_amounts: pd.Series, model_name: str) -> float:
    """
    Evaluates a model using the fraud capture score.
    """
    print(f"\n{model_name} Performance:")
    score = fraud_capture_score(y_true, y_pred_proba, transaction_amounts)
    print(f"Fraud Capture Score: {score:.2%}")
    return score

def evaluate_models_with_cv(X: pd.DataFrame, y: pd.Series, transaction_amounts: pd.Series, n_splits: int = 12):
    """
    Evaluates models using cross-validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    ml_scores, rule_based_scores = [], []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_splits}")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        amounts_val = transaction_amounts.iloc[val_index]

        ml_pipeline = FraudDetectionPipeline()
        ml_pipeline.fit(X_train, y_train)
        ml_predictions_proba = ml_pipeline.predict_proba(X_val)[:, 1]
        ml_scores.append(evaluate_model(y_val, ml_predictions_proba, amounts_val, "ML Model"))

        rule_based_predictions = rule_based_fraud_detection(X_val)
        rule_based_scores.append(evaluate_model(y_val, rule_based_predictions, amounts_val, "Rule-Based Model"))

    print("\nOverall Results:")
    print(f"ML Model Average Fraud Capture Score: {np.mean(ml_scores):.2%} (+/- {np.std(ml_scores):.2%})")
    print(f"Rule-Based Model Average Fraud Capture Score: {np.mean(rule_based_scores):.2%} (+/- {np.std(rule_based_scores):.2%})")
    print(f"Average Difference (ML - Rule-Based): {np.mean(np.array(ml_scores) - np.array(rule_based_scores)):.2%}")
