import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent))
from data_preprocessing.preprocessing import preprocessing
from eda_ftr_eng.EDA import eda
from eda_ftr_eng.feature_engineering import feature_engineering
from eda_ftr_eng.feature_selection import feature_selection

df_raw = pd.read_csv('Data/Raw/Job_Frauds.csv', encoding='latin-1')
preprocessed_df = preprocessing(df_raw)
#then rebalancing - tbc
eda_df = eda(preprocessed_df)
ftr_eng_df = feature_engineering(eda_df)
feature_selection_df = feature_selection(ftr_eng_df)

# modelling