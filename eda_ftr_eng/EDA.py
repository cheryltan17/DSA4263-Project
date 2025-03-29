import numpy as np
import pandas as pd

# Replace empty string, empty list as NaN
def replace_empty_values(x):
    if isinstance(x, str) and x.strip() == '':
        return np.nan
    elif isinstance(x, list) and len(x) == 0:
        return np.nan
    elif isinstance(x, (list, np.ndarray)) and len(x) == 0:
        return np.nan
    elif x is None:
        return np.nan
    else:
        return x

def eda(df):
    df = df[df['Country'] == 'us']
    df = df.drop(['State','Country'], axis=1)
    
    df = df.map(lambda x: replace_empty_values(x))

    # These columns are very significant when unspecified/specified
    df['Salary_Specified'] = df['Range_of_Salary'].notna()
    df['City_Specified'] = df['City'].notna()

    # Replace null values with 'Unspecified' category
    df = df.fillna("Unspecified")

    #Inpute with median to instead of replacing with 0 to prevent increasing bias
    df['Range_of_Salary'] = pd.to_numeric(df['Range_of_Salary'], errors='coerce')
    median_salary = df['Range_of_Salary'].median()
    df['Range_of_Salary'] = df['Range_of_Salary'].fillna(median_salary).astype(int)

    return df