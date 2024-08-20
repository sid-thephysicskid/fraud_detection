import pandas as pd

class DataLoader:
    def load_data(self, transactions_file: str = 'data/transactions_obf.csv', labels_file: str = 'data/labels_obf.csv') -> pd.DataFrame:
        """
        Loads transaction and label data from CSV files and merges them.
        """
        df = pd.read_csv(transactions_file, parse_dates=['transactionTime'])
        df['transactionTime'] = pd.to_datetime(df['transactionTime'], utc=True)
        # load and merge labels
        labels = pd.read_csv(labels_file)
        df['isFraud'] = df['eventId'].isin(labels['eventId']).astype(int)
        print("Data loaded. Shape:", df.shape)
        return df

class DataPreprocessor:
    def preprocess_fraud_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial preprocessing on the fraud data.
        """
        df = df.copy()
        df['transaction_to_cash_ratio'] = df['transactionAmount'] / df['availableCash']
        df['merchantZip_missing'] = df['merchantZip'].isna().astype(int)
        df['merchantZip'] = df['merchantZip'].fillna('Unknown')
        return df.drop(columns=['eventId'])
