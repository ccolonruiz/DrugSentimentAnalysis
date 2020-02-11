
# coding: utf-8

import pandas as pd
import numpy as np
import collections
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as soup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

class UDataset:
    def __init__(self, train=None, dev=None):
        self.__train = train
        self.__dev = dev
    
    def make_dev_split(self, dev_split=0.2):
        self.__train, self.__dev = np.split(self.__train, [int((1-dev_split)*len(self.__train)),]) 
    def get_train(self):
        return self.__train
    def get_dev(self):
        return self.__dev

class Dataset:
    def __init__(self, train=None, test=None, dev=None, label_name = None):
        self.__train = train
        self.__test = test
        self.__dev = dev
        self.__label_name = label_name
        
    def load_csv(self, train_path, test_path, label_name, separator, columns=None, dev_path=None):
        self.__train = pd.read_csv(train_path, sep=separator, header=0, usecols=columns)
        self.__test = pd.read_csv(test_path, sep=separator, header=0, usecols=columns)
        if dev_path is not None:
            self.__dev = pd.read_csv(dev_path, sep=separator, header=0, usecols=columns)
        self.__label_name = label_name
        
    def load_pickle(self, train_path, test_path, label_name, dev_path=None):
        self.__train = pd.read_pickle(train_path)
        self.__test = pd.read_pickle(test_path)
        if dev_path is not None:
            self.__dev = pd.read_pickle(dev_path)
        self.__label_name = label_name

    def parse_data(self, parse_col_name, parse_type):
        # example: dataset.parse_data(parse_col_name='review', parse_type='html.parser')
        if not self.get_all():
            raise Exception("Try to load the dataset using load_* functions. train or test are NoneType")
        self.__set_all([df[parse_col_name].apply(lambda x: soup(x, parse_type).text) for df in self.get_all()], parse_col_name)
        
    def parse_to_word_list(self, parse_col_name=None, stop=False, alphanum_sep=False, blind_num=False, blind_character='#'):
        if stop:     
            self.__set_all([[clear_stop_words(word_tokenize(split_alphanum(x, alphanum_sep, blind_num, blind_character).lower())) for x in df[parse_col_name]] for df in self.get_all()], parse_col_name)
        else:
            self.__set_all([[word_tokenize(split_alphanum(x, alphanum_sep, blind_num, blind_character).lower()) for x in df[parse_col_name]] for df in self.get_all()], parse_col_name)
            
    def list_to_stemmer(self, stem_col_name):
        stemmer = SnowballStemmer("english")
        self.__set_all([[[stemmer.stem(x) for x in sample] for sample in df[stem_col_name]] for df in self.get_all()], stem_col_name)
        
    def list_to_lemma(self, lemma_col_name):
        lemmatizer = WordNetLemmatizer()
        self.__set_all([[[lemmatizer.lemmatize(x) for x in sample] for sample in df[lemma_col_name]] for df in self.get_all()], lemma_col_name)

    def replace(self, parse_col_name=None, old=None, new=None):
        self.__set_all([[replace_character(sentence, old, new) for sentence in df[parse_col_name]] for df in self.get_all()], parse_col_name)
        
    def save_csv(self, train_path, test_path, separator, dev_path=None):
        if self.__train is None or self.__test is None:
            raise Exception("Train or Test are NoneType")
        self.__train.to_csv(train_path, sep=separator, index=False)
        self.__test.to_csv(test_path, sep=separator, index=False)
        if self.__dev is not None and dev_path is not None:
            self.__dev.to_csv(dev_path, sep=separator, index=False)
        
    def save_pickle(self, train_path, test_path, dev_path=None):
        if self.__train is None or self.__test is None:
            raise Exception("Train or Test are NoneType")
        self.__train.to_pickle(train_path)
        self.__test.to_pickle(test_path)
        if self.__dev is not None and dev_path is not None:
            self.__dev.to_pickle(dev_path)
        
    def make_dev_split(self, dev_split=0.2):
        if self.__dev is None and self.__train is not None:
            self.__dev = pd.DataFrame(columns=self.__train.columns)
            Xtrain, Xdev, ytrain, ydev = train_test_split(self.get_train_x(), self.get_train_y(), test_size=dev_split, random_state=42)
            self.__train = pd.concat([Xtrain, ytrain], axis=1).reset_index(drop=True)
            self.__dev = pd.concat([Xdev, ydev], axis=1).reset_index(drop=True)
        else:
            raise Exception("self.__train is None or self.__dev is already set")
        
    def classes_distribution(self, condition='columns'):
        elements = [x[eval("x."+condition)] for x in self.get_all()]
        tuples = [list(zip(*[[x, y] for x, y in df[self.__label_name].value_counts().items()])) for df in elements]
        sets = ['Entrenamiento','Test','Validación']
        colors = ['goldenrod', 'skyblue', 'lavender']

        plt.figure(figsize=(15,4))
        for i in range(len(tuples)):
            plt.subplot(1,len(tuples),i+1)
            plt.yticks(tuples[i][0])
            for a,b in zip(tuples[i][0], tuples[i][1]):
                plt.text(b, a, str(b))
            plt.xlabel('Número de comentarios')
            plt.ylabel('Puntuación')
            plt.title("Distribución de clases en dataset de "+sets[i])
            plt.barh(tuples[i][0], tuples[i][1], color=colors[i])
        
        if condition!='columns':
            num = sum([len(x) for x in elements])
            den = sum([len(x) for x in self.get_all()])
            return num/den
               
    def text_len_distribution(self, col_name, plot=False):    
        len_list = [len(x) for df in self.get_all() for x in df[col_name]]
        if plot:
            data = np.asarray(len_list)
            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            values, base = np.histogram(data, bins=(np.max(data)-np.min(data)), density=True)
            plt.title("Probability density function")
            plt.plot(base[:-1], values, c='green')
            plt.subplot(1,2,2)
            cumulative = np.cumsum(values)
            plt.title("Cumulative distribution function")
            plt.plot(base[:-1], cumulative, c='blue')
            plt.show()
            print("max len of texts: "+str(max(len_list))+"\nmin len of texts: "+str(min(len_list)))
        else:
            return max(len_list), min(len_list)
    
    def __set_all(self, newdata, column):
        self.__train[column] = newdata[0]
        self.__test[column] = newdata[1]
        if len(newdata)>2:
            self.__dev[column] = newdata[2]
            
    def set_column(self, newdata, column):
        self.__set_all(newdata, column)

    def get_all(self):
        return [i for i in [self.__train, self.__test, self.__dev] if i is not None]
    def get_train(self):
        return self.__train
    def get_test(self):
        return self.__test
    def get_dev(self):
        return self.__dev
    def get_train_x(self):
        return self.__train.loc[:, self.__train.columns != self.__label_name]
    def get_train_y(self):
        return self.__train[self.__label_name]
    def get_test_x(self):
        return self.__test.loc[:, self.__test.columns != self.__label_name]
    def get_test_y(self):
        return self.__test[self.__label_name]
    def get_dev_x(self):
        return self.__dev.loc[:, self.__dev.columns != self.__label_name]
    def get_dev_y(self):
        return self.__dev[self.__label_name]
    def get_label_name(self):
        return self.__label_name
    
class matrix_printer:
    def __init__(self, array, rows, columns):
        self.__array = array
        self.__index = rows
        self.__columns = columns
    def show(self):
        df_cm = pd.DataFrame(self.__array, index = self.__index,
        columns = [i for i in self.__columns])
        plt.figure(figsize = (10,6))
        sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)

def clear_stop_words(words):
    stops = set(stopwords.words('english'))
    return [w for w in words if not w in stops]

def replace_character(sentence, old=None, new=None):   
    return [new if word==old else word for word in sentence]

def split_alphanum(text, flag=True, blind_numbers=False, blind_character='#'):
    if flag:
        text = re.sub("[^\w\d,]", " ", text)
        text = re.sub("([0-9])([A-Za-z])", r"\1 \2", text)
        text = re.sub("([A-Za-z])([0-9])", r"\1 \2", text)
    if blind_numbers:
        text = re.sub("[0-9]+", blind_character, text)
    return(text)
                
