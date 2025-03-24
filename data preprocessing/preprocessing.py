## download packages
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')

## import libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


## read the data
df_raw = pd.read_csv('Data/Raw/Job_Frauds.csv', encoding='latin-1')


## split 'Job Location' into 'Country', 'State', 'City'
df = df_raw
split_location = df['Job Location'].str.split(', ', expand=True)
df['Country'] = split_location[0]  
df['State'] = split_location[1] 
df['City'] = split_location[2] 
df = df.drop(columns=['Job Location'])

## convert strings to lower case
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

## remove punctuations and symbols
df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)

## tokenize, remove stop words, and lemmatize for columns 'Profile', 'Job_Description', 'Requirements', 'Job_Benefits'
# initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()    
cols = ['Profile', 'Job_Description', 'Requirements', 'Job_Benefits']

for col in cols:
    # apply tokenization 
    df[col] = df[col].apply(lambda x: word_tokenize(x) if isinstance(x, str) and x.strip() != '' else [])
    # remove stop words
    df[col] = df[col].apply(lambda x: [word for word in x if word not in stop_words])
    # apply lemmatization
    df[col] = df[col].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

print(df)

## save as csv
#df.to_csv('Data/Processed/processed_df.csv', index=False)