import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess(path=None):
    if path is None:
        # Always resolve path from project root
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'creditcard.csv')
    
    print(f"📄 Loading dataset from: {path}")
    df = pd.read_csv(path)
    df['Amount_Norm'] = StandardScaler().fit_transform(df[['Amount']])
    df = df.drop(['Time', 'Amount'], axis=1)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y
