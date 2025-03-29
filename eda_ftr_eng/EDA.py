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
    df = df.map(lambda x: replace_empty_values(x))

    # Replace null values with 'Unspecified' category
    df = df.fillna("Unspecified")
    return df