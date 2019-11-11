# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from nltk.stem.wordnet import WordNetLemmatizer
import re
lemma = WordNetLemmatizer()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

def lemmatize(word):
    if len(word) < 4:
        return word
    word = lemma.lemmatize(word,"n")
    word = lemma.lemmatize(word,"v")
    return word


def preprocess(string):
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    string = string.split()
    for i in range(len(string)):
        string[i] = lemmatize(string[i])
    string = " ".join(string)
    return string

def preparing(data,test_size=0.2,random_state=2019):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:5], data.iloc[:,5], test_size=0.2, random_state=2019)
    X_train["question1"] = X_train["question1"].fillna("Nothing There").apply(preprocess)
    X_train["question2"] = X_train["question2"].fillna("Nothing There").apply(preprocess)
    X_test["question1"] = X_test["question1"].fillna("Nothing There").apply(preprocess)
    X_test["question2"] = X_test["question2"].fillna("Nothing There").apply(preprocess)
    
    return X_train, X_test, y_train, y_test


def get_word_list(X_train,MIN_WORD_OCCURRENCE=100):
    question_list = list(set(X_train["question1"].tolist() + X_train["question2"].tolist()))
    vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=MIN_WORD_OCCURRENCE)
    vectorizer.fit(question_list)
    words = vectorizer.get_feature_names()
    words.append("not_existed")
    
    return words


def get_embedding(words):
    embeddings = {}
    for line in open("glove.840B.300d.txt"):
        value = line.split(" ")
        if value[0] in words:
            embeddings[value[0]] = np.asarray(value[1:], dtype='float32')
    embeddings['not_existed'] = np.zeros(300)
    return embeddings

def is_numeric(s):
    return any(i.isdigit() for i in s)

def prepare(q,words,MAX_SEQUENCE_LENGTH,STOP_WORDS = set(stopwords.words('english'))):
    q2, not_existed, nums, flag= [], [],[], True
    for w in q.split():
        if w in words:
            q2 += [w]
            flag = True
        elif w not in STOP_WORDS:
            if flag:
                q2 += ["not_existed"] 
                flag = False
            if is_numeric(w):
                nums += [w] 
            else:
                not_existed += [w] 
        else:
            flag = True
        if len(q2) == MAX_SEQUENCE_LENGTH:
            break
    q2 = " ".join(q2)
    return q2, set(not_existed), set(nums)



def extract_features(df,words):
    q1s = np.array([""] * len(df), dtype=object)
    q2s = np.array([""] * len(df), dtype=object)
    features = np.zeros((len(df), 4))
    for i, (q1, q2) in enumerate(list(zip(df["question1"], df["question2"]))):
        q1s[i], non1, num1 = prepare(q1,words,30)
        q2s[i], non2, num2 = prepare(q2,words,30)
        features[i, 0] = len(non1.intersection(non2))
        features[i, 1] = len(non1.union(non2))
        features[i, 2] = len(num1.intersection(num2))
        features[i, 3] = len(num1.union(num2))

    return q1s, q2s, features

def final_prepare(q1s,q2s, tokenizer,MAX_SEQUENCE_LENGTH=30):
    data1 = pad_sequences(tokenizer.texts_to_sequences(q1s), maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    data2 = pad_sequences(tokenizer.texts_to_sequences(q2s), maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    
    return data1, data2 







