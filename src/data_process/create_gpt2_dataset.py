# define arguments
import argparse
import pickle
import time
import nltk

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file")
parser.add_argument("-ip", "--input-list-of-pubs", nargs="+")
parser.add_argument("-o", "--output-file")
args = parser.parse_args()

with open(args.input_file, 'rb') as handle:
    pubs_df = pickle.load(handle)

print("\n#########################################\n")
print("#  list of publications to concatenate  #")
print("\n#########################################\n")

list_of_pubs = args.input_list_of_pubs
print(list_of_pubs)

print("\n\nLoading input data file...\n\n")

with open(args.input_file, 'rb') as handle:
    pubs_df = pickle.load(handle)

def corpus_to_sent(corpus):
    return nltk.tokenize.sent_tokenize(corpus)

text_file = open(args.output_file, "w")
print("\n\nProcessing and adding articles sentence by sentence...\n\n")
start = time.process_time()
num_of_lines = 0

for x in list_of_pubs:
    for article in pubs_df[x].values:
        sents = corpus_to_sent(article)
        if sents:
            for sent in sents:
                text_file.write(sent)
                text_file.write("\n")
                num_of_lines += 1
        text_file.write("\n<|endoftext|>\n")
        num_of_lines += 1
text_file.close()
print("Total time (in seconds):")
print(time.process_time()-start)

print("\n\nTotal Number of Lines: ",num_of_lines)