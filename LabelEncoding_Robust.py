#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def Label_Robust(df):
    # Case 4: Label Encoding & RobustScaling
    # Label Encoding
    le = LabelEncoder()  
    df['career'] = le.fit_transform(df['career'])

    # Copy not scaled Data
    not_scaled_data = df.copy()

    # Scaling with Robust
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)

    print("<<Result of Label & Robust>>")
    print(df.head(5))
    
    return not_scaled_data, df
