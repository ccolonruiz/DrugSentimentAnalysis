# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:30:12 2017

@author: isegura
"""
from __future__ import division

import keras.backend as K
from keras.preprocessing.text import text_to_word_sequence

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def scores(y_test, y_pred):
 
    #0 for false, 1 for true
    y_test=[int(row[1]) for row in y_test]
    y_pred=[int(row[1]) for row in y_pred]
    
   
    tp_T=0
    tp_F=0

    fp_T=0
    fp_F=0

    fn_T=0
    fn_F=0

    for i,y in enumerate(y_pred):
        if (y_test[i]==1 and y==1):
            tp_T=tp_T+1
        elif (y_test[i]==0 and y==0):
            tp_F=tp_F+1
        elif (y_test[i]==0 and y==1):
            fp_T=fp_T+1
            fn_F=fn_F+1
        elif (y_test[i]==1 and y==0):
            fn_T=fn_T+1
            fp_F=fp_F+1

    print('true positives',tp_T,tp_F)
    print('false positives',fp_T,fp_F)
    print('false negatives',fn_T,fn_F)
    

    try:        
        precision_T= tp_T / (tp_T+fp_T)       
    except:
        precision_T=0

    precision_F= tp_F / (tp_F+fp_F)       
    print('precision',round(precision_T,2),round(precision_F,2))

    

    recall_T= tp_T / (tp_T+fn_T)       
    recall_F= tp_F / (tp_F+fn_F)  

    print('recall',round(recall_T,2),round(recall_F,2))

    try:
        f1_T = 2 * (precision_T * recall_T) / (precision_T + recall_T)
    except:
        f1_T = 0

    try:
        f1_F = 2 * (precision_F * recall_F) / (precision_F + recall_F)
    except:
        f1_F = 0

    
    print('f1',round(f1_T,2),round(f1_F,2))
    
    scores_T=[precision_T,recall_T,f1_T]
    print(scores_T)
    scores_F=[precision_F,recall_F,f1_F]
    print(scores_F)
    return scores_T,scores_F




import re

def replaceAccents(s):
    if len(s.strip())>0:
        s=re.sub('á', 'a', s)
        s=re.sub('é', 'e', s)
        s=re.sub('í', 'i', s)
        s=re.sub('ó', 'o', s)
        s=re.sub('ú', 'u', s)
        
        s=re.sub('à', 'a', s)
        s=re.sub('è', 'e', s)
        s=re.sub('ì', 'i', s)
        s=re.sub('ò', 'o', s)
        s=re.sub('ù', 'u', s)
        
        s=re.sub('Á', 'A', s)
        s=re.sub('É', 'E', s)
        s=re.sub('Í', 'I', s)
        s=re.sub('Ó', 'O', s)
        s=re.sub('Ú', 'U', s)
        
        s=re.sub('À', 'A', s)
        s=re.sub('È', 'E', s)
        s=re.sub('Ì', 'I', s)
        s=re.sub('Ò', 'O', s)
        s=re.sub('Ù', 'U', s)
    return s
    
def replaceNumbers(text):
    
    
    text=re.sub(r'\s+', ' ', text)
    text=re.sub(r'\d+\-\d+(\-\d+)*', 'DOSAGE', text)
    text=re.sub(r'-\d+\,*\d*', 'DIGIT', text)
    text=re.sub(r'\d+\,*\d*', 'DIGIT', text)
    text=re.sub(r'-\d+\.*\d*', 'DIGIT', text)
    text=re.sub(r'\d+\.*\d*', 'DIGIT', text)
    text=re.sub(r'-\d+', 'DIGIT', text)
    text=re.sub(r'\d+', 'DIGIT', text)
    
    return text



def clean(text):
    "'clean txt"
    text=replaceAccents(text)
    text=replaceNumbers(text)
    text=re.sub(r'\W+', ' ', text)#remove any non alphabetic character
    text=re.sub(r'\s+', ' ', text) #replace several whitespace by one
    text=text.strip()
    return text

def cleancollection(texts):
    aux=[]
    for text in texts:
        text=clean(text)
        aux.append(text)
    return aux

#Remove accents, replace numbers por digi label, remove any non alphanumeric character
def preprocess(text):
    "'clean txt"
    text=replaceAccents(text)
    text=replaceNumbers(text)
    sentences=text.split('.')
    lstS=[]
    for s in sentences:
        s=re.sub(r'\W+', ' ', s)
        s=re.sub(r'\s+', ' ', s)
        s=s.strip()
        lstS.append(s)
    return lstS


#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('spanish')

from nltk.stem import SnowballStemmer

def tokenizer_stem_nostop(text):
    stemmer = SnowballStemmer('spanish')
    return [stemmer.stem(w) for w in re.split('\s+', text.strip()) if w not in stop and re.match('[a-zA-Z]+', w)]

def tokenizer_stem(text):
    stemmer = SnowballStemmer('spanish')
    return [stemmer.stem(word) for word in re.split('\s+', text.strip())]
    
    
def maxLength(texts):
    maxLen=0
    for x in texts:
        tokens=text_to_word_sequence(x)
        length=len(tokens)
        if maxLen<length:
            maxLen=length
    return maxLen
    
    
def avgLength(texts):
    num=len(texts)
    sumLen=0
    for x in texts:
        tokens=text_to_word_sequence(x)
        #print(len(tokens))
        sumLen=sumLen+len(tokens)
        
    avg=sumLen//num
    return avg
