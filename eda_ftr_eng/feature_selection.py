from collections import Counter 
from sklearn.feature_selection import * 
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel 
from sklearn.linear_model import LogisticRegression, Lasso 
from sklearn.svm import LinearSVC 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def convert_scores_to_df(features: list[str], scores: list[float]) -> pd.DataFrame: 
    """Convert the feature & scores list to dataframe and round of the score to 3 decimal places 
 
    Args: 
        features (list[str]): Feature names 
        scores (list[float]): Score for respective features 
 
    Returns: 
        score_df (pd.Dataframe) 
    """ 
    return pd.DataFrame({'features':features,'scores':scores}).sort_values(by='scores',ascending=False).reset_index(drop=True).round(3)

def select_k_best(features:pd.DataFrame, labels:pd.DataFrame, score_func: str = 'f_classif', top_k:int = 20)-> dict[str, any]: 
    """Selects the top K features based on score function 
 
    Args: 
        features (pd.DataFrame): Feature Dataframe (X) 
        labels (pd.DataFrame): Labels Dataframe (Y) 
        score_func (str, optional): The scoring function to use for feature selection. Defaults to 'f_classif'. 
        top_k (int, optional): Top K Features to return from Features Dataframe. Defaults to 20. 
 
    Returns: 
        top_k_features (dict[str, any]): _description_ 
    """ 
    score_func = score_func.lower() 
    if score_func == 'f_classif': model = SelectKBest(score_func=f_classif, k=top_k) 
    if score_func == 'mutual_info_classif': model = SelectKBest(score_func=mutual_info_classif, k=top_k) 
    if score_func == 'chi2': model = SelectKBest(score_func=chi2, k=top_k) 
 
    model.fit(features, labels)  
    score_df = convert_scores_to_df (features.columns, model.scores_)    
    top_k_features = score_df.head(top_k)['features'].values 
    return { 
        'method': score_func, 
        'top_k_features': top_k_features, 
        'score_df': score_df.head(top_k) 
        } 

def select_k_best_chi2(features: pd.DataFrame, labels: pd.DataFrame, top_k: int = 20)-> dict[str, any]: 
    """ 
    Pros: Simple and fast; effective for categorical data 
    Cons: May not be suitable for purely numerical features 
    """ 
    return select_k_best(features=features, labels=labels, top_k=top_k, score_func='chi2') 

def select_random_forest(features: pd.DataFrame, labels: pd.DataFrame, top_k: int = 20)-> dict[str, any]: 
    """ 
    Selects the top K features using Random Forest feature importance. 
 
    Pros: Robust to overfitting; handles high-dimensional data 
    Cons: Computationally expensive; may not be as interpretable 
 
    Args: 
        features (pd.DataFrame): Feature Dataframe (X). 
        labels (pd.DataFrame): Labels Dataframe (Y). 
        top_k (int, optional): Top K Features to return from Features Dataframe. Defaults to 20. 
 
    Returns: 
        dict[str, any]: List of the top K feature names. 
    """ 
    model = RandomForestClassifier() 
    model.fit(features, labels) 
    # selector = SelectFromModel(model, max_features=top_k, prefit=True) 
    score_df = convert_scores_to_df (features.columns, model.feature_importances_)   
    top_k_features = score_df.head(top_k)['features'].values 
    return { 
        'method': 'RandomForest', 
        'top_k_features': top_k_features, 
        'score_df': score_df.head(top_k) 
        } 

def select_features(data, features, model_name, k):
    if model_name == "select_k_best": model = select_k_best
    elif model_name == "select_random_forest": model = select_random_forest
    elif model_name == "select_k_best_chi2": model= select_k_best_chi2
    else: print("No such model")
    
    model_results = model(data[features], data['Fraudulent'], top_k = k) #k: number of features selected by the model
    selected_features = model_results['top_k_features']
    top_k_df = model_results['score_df']
    return top_k_df

def normalise_score_df(df, scoring_name):
    scaler = MinMaxScaler()
    df = df.copy()  # Avoid modifying original
    df['scores'] = scaler.fit_transform(df[['scores']])
    return df

def feature_selection(df):
    columns_for_embedding = ['Job Title', 'Profile', 'Department', 'Job_Description', 'Requirements', 'Job_Benefits', 'Type_of_Industry', 'Operations','City']

    embed_cols = []
    for col in columns_for_embedding:
        embed_col_name = f"{col}_embed"
        embed_cols.append(embed_col_name)
    non_embed_df = df.drop(embed_cols, axis=1).reset_index(drop=True)

    features = non_embed_df.drop(['Fraudulent'], axis=1).columns

    score_df_k_best = select_features(non_embed_df, features, "select_k_best", 22)
    score_df_rf= select_features(non_embed_df, features, "select_random_forest", 22)

    #Chi2 works best for categorical variable, so we remove Range of Salary
    features = non_embed_df.drop(['Fraudulent', 'Range_of_Salary'], axis=1).columns
    score_df_chi2= select_features(non_embed_df, features, "select_k_best_chi2", 22)

    # ftr_selection_df_score['feature'] = score_df_k_best['features']
    score_df_k_best_normalised = normalise_score_df(score_df_k_best, "score_df_k_best")
    score_df_rf_normalised = normalise_score_df(score_df_rf, "score_df_rf")
    score_df_chi2_normalised = normalise_score_df(score_df_chi2, "score_df_chi2")

    # Merge all DataFrames on 'features' column
    combined_scores = (
        score_df_k_best_normalised[['features', 'scores']]
        .merge(score_df_rf_normalised[['features', 'scores']], 
            on='features', 
            how='outer', 
            suffixes=('_k_best', '_rf'))
        .merge(score_df_chi2_normalised[['features', 'scores']], 
            on='features', 
            how='outer')
        .rename(columns={'scores': 'scores_chi2'})
    )

    combined_scores['mean_score'] = combined_scores[['scores_k_best', 'scores_rf', 'scores_chi2']].mean(axis=1, skipna=True)

    # Sort by k_best scores (or any other method)
    combined_scores = combined_scores.sort_values('mean_score', ascending=False)

    #force reset the index
    combined_scores.index = range(len(combined_scores))

    #get the features with mean scores of > 0.02
    combined_scores = combined_scores[combined_scores['mean_score'] > 0.02]
    selected_features_non_embed = combined_scores['features'].to_list()
    selected_features = selected_features_non_embed + embed_cols
    result_df = df[selected_features] 

    return result_df