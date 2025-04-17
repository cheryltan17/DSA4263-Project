from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def simplify_qualification(q):
    if pd.isna(q) or q in ["unspecified", "other"]:
        return "unspecified"
    elif q in ["high school or equivalent", "some high school coursework", "vocational hs diploma"]:
        return "high school"
    elif q in ["vocational", "vocational degree", "certification", "professional"]:
        return "vocational / certification/ professional"
    elif q == "some college coursework completed":
        return "some college"
    elif q == "associate degree":
        return "associate degree"
    elif q == "bachelors degree":
        return "bachelor's degree"
    elif q == "masters degree":
        return "master's degree"
    elif q == "doctorate":
        return "doctorate"
    else:
        return "unspecified"

def simplify_employment_type(q):
    if pd.isna(q) or q in ["other", "unknown"]:
        return "unspecified"
    else:
        return q
    
def simplify_experience(q):
    if pd.isna(q) or q in ["not applicable", "unknown"]:
        return "unspecified"
    else:
        return q

def feature_engineering(df):
    df['Qualification'] = df['Qualification'].apply(simplify_qualification)
    df['Type_of_Employment'] = df['Type_of_Employment'].apply(simplify_employment_type)
    df['Experience'] = df['Experience'].apply(simplify_experience)

    columns_for_embedding = ['Job Title', 'Profile', 'Department', 'Job_Description', 'Requirements', 'Job_Benefits', 'Type_of_Industry', 'Operations','City']
    categorical_columns = ['Type_of_Employment','Experience', 'Qualification' ]
    
    #Encoding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = df.reset_index(drop=True) #need to do this
    for col in columns_for_embedding:
        print(f"Embedding column: {col}")
        df[f"{col}_embed"] = list(model.encode(df[col].fillna("")))
    df_aft_str_embed = df.drop(columns_for_embedding, axis=1)

    # One-Hot Encode
    df_aft_enc = pd.get_dummies(df_aft_str_embed, columns=categorical_columns, drop_first=True)   
    
    # Convert to integer for modelling
    for col in df_aft_enc.columns:
        if df_aft_enc[col].dtype == 'bool':
            df_aft_enc[col] = df_aft_enc[col].astype(int)

    #Scale Range of Salary
    scaler = MinMaxScaler()
    df_aft_enc['Range_of_Salary'] = scaler.fit_transform(df_aft_enc[['Range_of_Salary']])

    return df_aft_enc


