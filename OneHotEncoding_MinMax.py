#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def OneHot_MinMax(df):
    # Case 1: One-hot Encoding & MinMax
    # One-hot Encoding by get_dummies function in pandas lib
    df = pd.get_dummies(df, columns = ['career'])

    # Copy not scaled Data
    not_scaled_data = df.copy()

    # Scaling with MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)

    print("<<Result of OneHot & MinMax>>")
    print(df.head(5))
    
    return not_scaled_data, df

    
