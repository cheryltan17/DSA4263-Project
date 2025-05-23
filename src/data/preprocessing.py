## download packages
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')

## import libraries
import pandas as pd
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import MinMaxScaler

## function to replace accents (eg. áéíóú -> aeiou)
def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')

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
    
def preprocessing(df):
    ## split 'Job Location' into 'Country', 'State', 'City'
    split_location = df['Job Location'].str.split(', ', expand=True)
    df['Country'] = split_location[0]  
    df['State'] = split_location[1] 
    df['City'] = split_location[2] 
    df = df.drop(columns=['Job Location'])

    ## replaced NaN with 'unknown'
    text_cols = ['Job Title','Profile', 'Job_Description', 'Requirements', 'Job_Benefits','Type_of_Employment','Experience','Qualification',
            'Department', 'Type_of_Industry', 'Operations','Country','State','City']
    df[text_cols] = df[text_cols].fillna('unknown')

    ## convert strings to lower case
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    ## remove punctuations and symbols
    df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)

    ## replace accents
    for col in text_cols:
        # replace accent
        df[col] = df[col].apply(lambda x: [strip_accents(x)])
        df[col] = df[col].apply(lambda x: " ".join(x))


    ## tokenize, remove stop words, and lemmatize for columns 'Profile', 'Job_Description', 'Requirements', 'Job_Benefits','Job Title',
    ## 'Department', 'Type_of_Industry', 'Operations
    # initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()    
    cols = ['Profile', 'Job_Description', 'Requirements', 'Job_Benefits','Job Title', 'Department', 'Type_of_Industry', 'Operations']

    for col in cols:
        # apply tokenization 
        df[col] = df[col].apply(lambda x: word_tokenize(x) if isinstance(x, str) and x.strip() != '' else [])
        # remove stop words
        df[col] = df[col].apply(lambda x: [word for word in x if word not in stop_words])
        # apply lemmatization
        df[col] = df[col].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    for col in cols:
        df[col] = df[col].apply(lambda x: " ".join(x))
    return df