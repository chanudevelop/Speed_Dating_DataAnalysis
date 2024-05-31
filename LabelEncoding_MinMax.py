#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def Label_MinMax(df):
    # Case 3: LabelEncoding & MinMax
    # Label Encoding
    le = LabelEncoder()  
    df['career'] = le.fit_transform(df['career'])
    
    # Copy not scaled Data
    not_scaled_data = df.copy()

    # Scaling with MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)

    print("<<Result of Label  & MinMax>>")
    print(df.head(5))
    
    return not_scaled_data, df

    
