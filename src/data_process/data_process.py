# arg parse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", type=str)
parser.add_argument("-o", "--output-file", type=str)
args = parser.parse_args()

# initial imports
import pandas as pd
import numpy as np
import os
import re
np.random.seed(0)

# textblob sentiment analysis
from textblob import TextBlob

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

df = pd.read_csv(args.input_file)
print("\n\n")
print("printing the list of publications in this file")
pubs = np.unique(df.publication)
print(pubs)

print("\n\n")
print("Number of articles per publications")
total_count = len(df)
for x in pubs:
    print(x)
    n = df[df.publication == x].shape[0]
    print("\tNumber of Articles:",n)
    print("\tFraction of the dataset:",round(n/total_count*100,2))


print("\n\n")
print("Separating articles by publications...")
pubs_df = {}
for x in pubs:
    pubs_df[x] = df[df.publication == x].content

with open(args.output_file, "wb") as handle:
    pickle.dump(pubs_df, handle, protocol = pickle.HIGHEST_PROTOCOL)

print("\n...\n")
print("Finished!")
