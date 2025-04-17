from collections import Counter 
from sklearn.feature_selection import * 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import *
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def convert_scores_to_df(features: list[str], scores: list[float]) -> pd.DataFrame: 
    """Convert the feature & scores list to dataframe and round of the score to 3 decimal places 
 
    Args: 
        features (list[str]): Feature names 
        scores (list[float]): Score for respective features 
 
    Returns: 
        score_df (pd.Dataframe) 
    """ 
    return pd.DataFrame({'features':features,'scores':scores}).sort_values(by='scores',ascending=False).reset_index(drop=True).round(3)

def select_rfe_with_class_weights(features: pd.DataFrame, labels: pd.DataFrame, top_k: int = 20)-> dict[str, any]:
    """
    Selects the top K features using Recursive Feature Elimination (RFE) with class weights adjusted.

    Pros: Addresses class imbalance; interpretable
    Cons: Computationally intensive for large datasets

    Args:
        features (pd.DataFrame): Feature Dataframe (X).
        labels (pd.DataFrame): Labels Dataframe (Y).
        top_k (int, optional): Top K Features to return from Features Dataframe. Defaults to 20.

    Returns:
       dict[str, any]: List of the top K feature names.
    """
    model = LogisticRegression(class_weight='balanced')
    rfe = RFE(model, n_features_to_select=top_k)
    rfe.fit(features, labels)
    rank_df = convert_scores_to_df (features.columns, rfe.ranking_)
    #The feature ranking, such that ranking_[i] corresponds to the ranking position of the i-th feature. Selected (i.e., estimated best) features are assigned rank 1.
    top_k_df = rank_df[rank_df['scores'] == 1]
    top_k_features = top_k_df['features'].values
    return {
        'method': 'Recursive Feature Elimination (rankings) - Class Balanced',
        'top_k_features': top_k_features,
        'score_df': top_k_df
        }

def select_features(data, features, model_name, k):
    if model_name == "select_rfe_with_class_weights": model = select_rfe_with_class_weights
    else: print("No such model")

    model_results = model(data[features], data['Fraudulent'], top_k = k) #k: number of features selected by the model
    selected_features = model_results['top_k_features']
    top_k_df = model_results['score_df']
    return top_k_df, selected_features

def feature_selection(df):
    columns_for_embedding = ['Job Title', 'Profile', 'Department', 'Job_Description', 'Requirements', 'Job_Benefits', 'Type_of_Industry', 'Operations','City']

    embed_cols = []
    for col in columns_for_embedding:
        embed_col_name = f"{col}_embed"
        embed_cols.append(embed_col_name)
    non_embed_df = df.drop(embed_cols, axis=1).reset_index(drop=True)

    data = non_embed_df
    y = data['Fraudulent']
    x_vif = data.drop(['Fraudulent'], axis=1)
    thres = 5

    #VIF
    while True:
        Cols = range(x_vif.shape[1])
        vif = np.array([variance_inflation_factor(x_vif.values, i) for i in Cols])

        if all(vif < thres):
            break
        else:
            max_vif_idx = np.argmax(vif)
            col_to_drop = x_vif.columns[max_vif_idx]
            print(f"Dropping column '{col_to_drop}' with VIF = {vif[max_vif_idx]:.2f}")
            x_vif = x_vif.drop(columns=[col_to_drop])

    x_vif['Fraudulent'] = y

    #RFE
    data = x_vif
    y = data['Fraudulent']
    X = data.drop(columns=['Fraudulent'])


    model = RandomForestClassifier(random_state=42, class_weight='balanced')

    k_range = range(1, X.shape[1])
    cv_scores = []
    cv_scores_per_k = []

    for k in k_range:
        rfe = RFE(estimator=model, n_features_to_select=k)
        X_rfe = rfe.fit_transform(X, y)

        score = cross_val_score(model, X_rfe, y, cv=5, scoring='f1').mean()
        cv_scores.append(score)
        cv_scores_per_k.append(score/k)
    
    # Convert range to list if it's a range object
    k_values = list(k_range)

    # Detect elbow (knee) point
    knee = KneeLocator(k_values, cv_scores_per_k, curve="convex", direction="decreasing")

    print("Optimal number of features (elbow point):", knee.knee)

    #Select the number of features based on the elbow
    features = x_vif.drop(['Fraudulent'], axis=1).columns
    score_df_k_best, selected_ftrs_k = feature_selection(x_vif, features, "select_rfe_with_class_weights", knee.knee)

    embed_cols = []
    for col in columns_for_embedding:
        embed_col_name = f"{col}_embed"
        embed_cols.append(embed_col_name)

    selected_features = selected_ftrs_k.tolist() + embed_cols + ["Fraudulent"]
    result_df = df[selected_features]
    return result_df