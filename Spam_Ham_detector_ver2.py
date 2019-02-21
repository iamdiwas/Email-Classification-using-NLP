# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:16:45 2019

@author: e67967
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
total_data = pd.read_csv("W:/Diwas/Python_projects/NLP_Codes/Spam_Ham_detector/spam.csv",encoding = 'latin-1')

working_data = total_data.iloc[:,0:2] ## using only useful data and remove all the junk of the text ##
working_data.head(n=10)
working_data.isna().sum() ## check and drop all the null values ##
working_data = working_data.dropna()
raw_labels = pd.DataFrame(working_data.iloc[:,0])
encoder = LabelEncoder()
y_enc = encoder.fit_transform(raw_labels) ## Encoded spamand ham labels ##


#### data visualization ####

spam_text = list(working_data[working_data['v1'] == "spam"]['v2'])
spam_text = ''.join(spam_text)

spam_wordcloud = WordCloud(width=1024,height=1024).generate(spam_text)
plt.imshow(spam_wordcloud)
plt.figure(figsize=(10,10))
plt.show()

ham_text = list(working_data[working_data['v1'] == "ham"]['v2'])
ham_text = ''.join(ham_text)

ham_wordcloud = WordCloud(width=1024,height=1024).generate(ham_text)
plt.imshow(ham_wordcloud)
plt.show()

#################### Data Feature Engineering ########################  
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

raw_text = pd.DataFrame(working_data['v2']) ## Use only text data  coloumn ##
def Text_Cleanning(text):
    
    word_tokens = word_tokenize(text)
    word_tokens = [word.lower() for word in word_tokens]
    
    str_punctuation = str.maketrans('','',string.punctuation) ## collecting all the punctuation marks ##
    punctuation_rem = [word.translate(str_punctuation) for word in word_tokens] ## remove all the puncuation ##
    
        
    text_cleaned = [word for word in punctuation_rem if word.isalpha()] ## after all punctuation removed, remove all the non text terms ##
    text_cleaned = " ".join(text_cleaned) ## join all the tokenized terms together ##
    return text_cleaned

## Collecting all the cleaned text ##
text_cleaned = []

for index,rows in raw_text.iterrows():
    
    text_cleaned.append(Text_Cleanning(rows['v2']))                    
## convert all the dataframe to series format ##
text_cleaned = pd.DataFrame(text_cleaned)
text_cleaned.columns = ['messages']                   
text_cleaned = pd.Series(text_cleaned['messages']) 

## Now, start the ML data training and testing ##
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
#import xgboost as xgb
tfidf = TfidfVectorizer(ngram_range=(1,2)) ## using bigrams and unigrams text words combination to train the system ##
x_features = tfidf.fit_transform(text_cleaned)

x_train,x_test,y_train,y_test = train_test_split(x_features, y_enc , test_size = 0.25)
################## Boosting Classifier ###################
model = AdaBoostClassifier()
model.fit(x_train,y_train)
################# SVM Classifier #########
#svc_classifier = SVC()
#svc_classifier.fit(x_train,y_train)
### xgboost model ###
#xgb_model = xgb.train(x_train,y_train)
y_pred = model.predict(x_test)
print(f1_score(y_test,y_pred))

pd.DataFrame(confusion_matrix(y_test,y_pred), index = [['actual','actual'],['spam','ham']],columns = [['predicted','predicted'],['spam','ham']])






                 
                    