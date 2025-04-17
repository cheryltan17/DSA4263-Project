import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix, roc_auc_score, auc,f1_score,precision_score,recall_score

## convert embedded columns to numpy arrays
def convert_to_array(s):
    # Clean the string
    cleaned_str = s.strip('[]').replace('\n', ' ').replace('  ', ' ').strip()
    # Convert the cleaned string to a NumPy array
    return np.fromstring(cleaned_str, sep=' ')


## flatten embedded columns
def flatten(df):    
    embedded_cols = ["Job Title_embed","Profile_embed","Department_embed","Job_Description_embed","Requirements_embed",
                "Job_Benefits_embed","Type_of_Industry_embed","Operations_embed","City_embed"]
    categorical_feats = ["Comnpany_Logo","Qualification_master's degree","Qualification_vocational / certification/ professional",
                     "Experience_executive", "Qualification_doctorate"]
    X_text = np.hstack([np.vstack(df[col]) for col in embedded_cols]) 
    X_cat = df[categorical_feats].values
    return np.hstack([X_text, X_cat])

def do_pca(X_std):
    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    
    # Plot Explained Variance Ratio
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs. Number of Principal Components')
    plt.show()
    return X_pca
    

def resampling_method(method,X_train,y_train):
    if method == 'Undersampling':
        rus = RandomUnderSampler(random_state=4263)
        X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
    elif method == 'Oversampling':
        ros = RandomOverSampler(random_state=4263)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
    else:
        print('Invalid resampling method')
    resampled_class_counts = pd.Series(y_train_balanced).value_counts()
    print("Resampled class counts:")
    print(resampled_class_counts)
    return X_train_balanced, y_train_balanced

def evaluate_model(y_test,y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix for {model_name}')
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test,y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    performance_score = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    print(classification_report(y_test, y_pred))
    return performance_score

def auc_roc(model,model_name,X_test,y_test):
    pred_prob = model.predict_proba(X_test)
    auc_score = round(roc_auc_score(y_test, pred_prob[:,1]),3)
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    plt.plot(fpr, tpr, linestyle='--',color='orange', label ='ROC curve (area = %0.3f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    
    return print(f'AUC score: {auc_score}')