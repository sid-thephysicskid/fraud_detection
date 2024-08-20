## Project Overview

This project implements a machine learning solution to flag fraudulent transactions given the constraints of the business problem: aka, only 400 transactions can be reviewed per month, and the goal is to prevent the most  amount of fraud  value.

### Key Features:
- Time-based train-test split to simulate production conditions (last month used as test data)
- Custom feature engineering for fraud detection
- Machine learning model (XGBoost) for fraud prediction
- Rule-based fraud detection for comparison
- Cross-validation for robust model evaluation
- Performance evaluation using a custom fraud capture score

## Project Structure

```
├── data/
│   ├── transactions_obf.csv
│   └── labels_obf.csv
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_pipeline.py
│   └── evaluation.py
└── README.md
```

## Installation and Setup

1. Install required packages ussing requirements.txt:
   ```
   pip install -r requirements.txt
   ```

   Alternatively, if there are any clashes, the following should work:
   
   ```
   pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib
   ```
2. Ensure the data files are in the `data/` directory

## Usage

Run the main script:

```
python src/main.py
```

This will:
1. Load and preprocess the data
2. Perform cross-validation
3. Train the final model
4. Evaluate the model on the test set
5. Compare the ML model with a rule-based approach
6. Save the trained model

## Model Performance

The machine learning model significantly outperforms the rule-based approach:

- ML Model Fraud Capture Score: 90.52%
- Rule-Based Model Fraud Capture Score: 63.93%
- Improvement: 26.59%

Cross-validation results:
- ML Model Average Fraud Capture Score: 88.36% (+/- 3.89%)
- Rule-Based Model Average Fraud Capture Score: 77.51% (+/- 8.16%)
- Average Improvement: 10.86%

## Key Components

- `DataLoader`: Loads transaction and label data
- `DataPreprocessor`: Performs initial data preprocessing
- `CustomFeatureEngineer`: Implements domain-specific feature engineering
- `FraudDetectionPipeline`: Combines feature engineering, preprocessing, and model training
- `evaluate_models_with_cv`: Performs cross-validation for model evaluation
- `fraud_capture_score`: Custom metric for evaluating model performance

## Future Improvements

1. Feature importance analysis for better understanding of the model
2. Hyperparameter tuning for potentially improved performance
3. Exploration of other machine learning algorithms

