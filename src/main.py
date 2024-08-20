import numpy as np
from data_loader import DataLoader, DataPreprocessor
from evaluation import time_based_train_test_split, evaluate_models_with_cv, evaluate_model, rule_based_fraud_detection
from model_pipeline import FraudDetectionPipeline
from joblib import dump

def plot_feature_importance(shap_values, feature_names, output_file='feature_importance.png'):
    """
    plot feature importance based on SHAP values.
    """
    shap.summary_plot(shap_values, features=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Feature importance plot saved as {output_file}")

def main():
    # Load and preprocess data
    data_loader = DataLoader()
    df = data_loader.load_data('data/transactions_obf.csv', 'data/labels_obf.csv')
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_fraud_data(df)

    # split data based on time (simulating how it'd be in production)
    train_df, test_df = time_based_train_test_split(df)

    # Prepare features and target
    features = ['accountNumber', 'merchantId', 'merchantZip', 'mcc', 'merchantCountry', 'posEntryMode',
                'transactionAmount', 'availableCash', 'transaction_to_cash_ratio', 'transactionTime',
                'merchantZip_missing']

    X_train, y_train = train_df[features], train_df['isFraud']
    X_test, y_test = test_df[features], test_df['isFraud']

    # Perform cross-validation
    print("\n Let's first evaluate the ML model vs Rule based Algorithm\
     \n using stratified 12 fold Cross-Validation:")
    evaluate_models_with_cv(X_train, y_train, X_train['transactionAmount'], n_splits=12)

    # Final evaluation on test set
    print("\n Now the Final Evaluation on Test Set:")
    ml_pipeline = FraudDetectionPipeline()
    ml_pipeline.fit(X_train, y_train)
    ml_predictions_proba = ml_pipeline.predict_proba(X_test)[:, 1]
    ml_score = evaluate_model(y_test, ml_predictions_proba, X_test['transactionAmount'], "Machine Learning Model")

    rule_based_predictions = rule_based_fraud_detection(X_test)
    rule_based_score = evaluate_model(y_test, rule_based_predictions, X_test['transactionAmount'], "Rule-Based Model")

    print("\nFinal Comparison:")
    print(f"ML Model Fraud Capture Score: {ml_score:.2%}")
    print(f"Rule-Based Model Fraud Capture Score: {rule_based_score:.2%}")
    print(f"Difference (ML - Rule-Based): {(ml_score - rule_based_score):.2%}")
    dump(ml_pipeline, 'fraud_detection_model.joblib')
    print("\nModel saved as fraud_detection_model.joblib")


if __name__ == "__main__":
    main()
