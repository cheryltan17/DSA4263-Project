import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def oversampling(df):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(df.drop(columns=['Fraudulent']), df['Fraudulent'])
    df_oresampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_oresampled

def undersampling(df):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(df.drop(columns=['Fraudulent']), df['Fraudulent'])
    df_uresampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_uresampled