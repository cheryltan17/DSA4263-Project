import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent))
from data_preprocessing.preprocessing import preprocessing
from eda_ftr_eng.EDA import eda

df_raw = pd.read_csv('Data/Raw/Job_Frauds.csv', encoding='latin-1')
preprocessed_df = preprocessing(df_raw)
#then rebalancing - tbc
eda_df = eda(preprocessed_df)
# feature engineering
# feature selection
# modelling