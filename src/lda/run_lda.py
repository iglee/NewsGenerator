# arg parse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", type=str)
parser.add_argument("-p", "--publication", help="add publication name to be parsed by LDA. possible options: Atlantic, Breitbart, Business Insider, \
Buzzfeed News, CNN, Fox News, Guardian, NPR, National Review, New York Post, New York Times, Reuters, Talking Points Memo, Vox, Washington Post", type=str)
parser.add_argument("-o", "--output-file", type=str)
parser.add_argument("-n", "--num-topics", type=int, default=20)
parser.add_argument("-w", "--num-words", type=int, default=10)
args = parser.parse_args()

# initial imports
import pandas as pd
import numpy as np
import os
import re
import time
np.random.seed(0)

# LDA
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk
from nltk.corpus import stopwords

# for saving file
import pickle

with open(args.input_file, 'rb') as handle:
    pubs_df = pickle.load(handle)

print("Isolating a particular publication content")
print("\n"+args.publication+"\n")
pub_content = pubs_df[args.publication].reindex()
pub_content = pub_content.values

############################################################
#            Process Corpus with gensim for LDA            #
############################################################

stop_words = stopwords.words('english')
stop_words.extend(["news"])

def corpus_process(corpus):
    return nltk.tokenize.sent_tokenize(corpus)

def sentence_process(processed_corpus):
    tokenized_sentences = []
    for sentence in processed_corpus:
        tokenized_sentences.append(simple_preprocess(sentence, deacc=True))
    return tokenized_sentences

def process_entire_publication(pub):
    processed_pubs = []
    
    for article in pub:
        processed_corpus = corpus_process(article)
        tokenized_corpus = sentence_process(processed_corpus)
        processed_pubs.append(tokenized_corpus)
    
    return processed_pubs

def compose_corpus(processed_pub):
    corpus = []

    for articles in processed_pub:
        a = []
        for s in articles:
            for t in s:
                if t not in stop_words:
                    a.append(t)
        corpus.append(a)
    return corpus

print("parsing articles...")
start = time.process_time()
texts = compose_corpus(pub_content)
print("\n   total execution time in seconds: ")
print(time.process_time() - start)
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]
print("\nstart training gensim LDA...")
model_start = time.process_time()
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,\
                                           id2word=id2word,\
                                           num_topics=args.num_topics, \
                                           random_state=0,\
                                           update_every=1,\
                                           chunksize=100,\
                                           passes=10,\
                                           alpha='auto',\
                                           per_word_topics=True)
print("\n   total model training time in seconds: ")
print(time.process_time() - model_start)

print("\nprinting the topics...")
topics = lda_model.print_topics(num_words=args.num_words)
for topic in topics:
    print(str(topic))

print("Saving model...")
lda_model.save(args.output_file)
