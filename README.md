# Comparing Deep Learning architectures for Sentiment Analysis on Drug Reviews.

## Abstract

Since the turn of the century, as millions of user's opinions are available on the web, sentiment analysis has become one of the most fruitful research fields in Natural Language Processing (NLP). Research on sentiment analysis has covered a wide range of domains such as economy, polity, and medicine, among others. In the pharmaceutical field, automatic analysis of online user reviews allows for the analysis of large amounts of user's opinions and to obtain relevant information about the effectiveness and side effects of drugs, which could be used to improve pharmacovigilance systems. 
Throughout the years, approaches for sentiment analysis have progressed from simple rules to advanced machine learning techniques such as deep learning, which has become an emerging technology in many NLP tasks. 
Sentiment analysis is not oblivious to this success, and several systems based on deep learning have recently demonstrated their superiority over former methods, achieving state-of-the-art results on standard sentiment analysis datasets. However, prior work shows that very few attempts have been made to apply deep learning to sentiment analysis of drug reviews. Moreover, most previous research only dealt with the classification of two or three polarities (positive, negative and neutral). In this paper, we consider a more challenging task where each drug review is classified with an overall rating from 1 to 10. We present a benchmark comparison of various hybrid deep learning architectures, in addition to exploring the effect of adding different pre-trained word embeddings models. We also propose a novel architecture that consists of an LSTM followed by a CNN, where the CNN computes the feature maps after receiving the output hidden unit sequence from LSTM.
Experimental results demonstrate that the novel approach marginally outperforms the comparison models (69,28% of micro-F1).

## W2V Models

S. Pyysalo, F. Ginter, H. Moen, T. Salakoski, S. Ananiadou, Distributional semantics resources for biomedical text processing, Proceedings of Languages in Biology and Medicine

A. Nikfarjam, A. Sarker, K. O’Connor, R. Ginn, G. Gonzalez, Pharmacovigilance from social media: mining adverse drug reaction mentions using sequence labeling with word embedding cluster features, Journal of the American Medical Informatics Association 22 (3) (2015) 671–681.

Q. Li, S. Shah, X. Liu, A. Nourbakhsh, Data sets: Word embeddings learned from tweets and general data, in: Eleventh International AAAI Conference on Web and Social Media, 2017

## Dataset 

F. Gräßer, S. Kallumadi, H. Malberg, S. Zaunseder, Aspect-based sentiment analysis of drug reviews applying cross-domain and cross-data learning, in: Proceedings of the 2018 International Conference on Digital Health, ACM, 2018, pp. 121–125.
