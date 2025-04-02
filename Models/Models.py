import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix, roc_auc_score, auc

## convert embedded columns to numpy arrays
def convert_to_array(s):
    # Clean the string
    cleaned_str = s.strip('[]').replace('\n', ' ').replace('  ', ' ').strip()
    # Convert the cleaned string to a NumPy array
    return np.fromstring(cleaned_str, sep=' ')

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
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return cm

def auc_roc(model,model_name,X_test,y_test):
    pred_prob = model.predict_proba(X_test)
    auc_score = round(roc_auc_score(y_test, pred_prob[:,1]),3)
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    plt.plot(fpr, tpr, linestyle='--',color='orange', label ='ROC curve (area = %0.3f)' % auc_score)
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    
    return print(f'AUC score: {auc_score}')