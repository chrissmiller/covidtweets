
import pickle
import os

import pandas as pd
import numpy as np
import collections
from collections import Counter

import spacy
from spacy import displacy

import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import webtext
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import ADJ, ADV, NOUN, VERB
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from gensim import corpora, models

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('universal_tagset')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, help='Seed')
parser.add_argument('--per_file', default=100000, help='Number of tweets per file')

args = parser.parse_args()


'''
https://towardsdatascience.com/topic-modeling-of-2019-hr-tech-conference-twitter-d16cf75895b6
Data import
'''

np.random.seed(int(args.seed))

datafiles = ['data/2020-' + date for date in ['03-29', '03-30', '03-31', '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '04-07', '04-08', '04-09', '04-10', '04-11', '04-12', '04-13', '04-14', '04-15', '04-16']]

per_file = int(args.per_file)
covid_data = pd.read_csv(datafiles[0]).sample(n=per_file)
for file in datafiles[1:]:
    next_file = pd.read_csv(file)
    next_file = next_file.sample(n=per_file)
    covid_data = covid_data.append(next_file)

print(f"Evaluating {len(covid_data)} tweets.")

# We limit to english tweets to ensure topics are applicable to all tweets
covid_data_en = covid_data[covid_data.lang == 'en']
print(f"Found {len(covid_data_en)} English tweets.")


covid_data_en.created_at = pd.to_datetime(covid_data_en.created_at)
covid_data_en["dayofyear"] = covid_data_en.created_at.map(lambda x: x.dayofyear)

# Shuffle data
data = covid_data_en.sample(frac=1)
covid_data_en = None
covid_data = None
'''
SpaCy Named Entity recognition

Model trained on OntoNotes5 corpus which contains different genres of text in English, Chinese and Arabic associated with structural information.

In english the word breakdown of text is:
*   625k news
*   200k broadcast news
*   200k broadcast conversation
*   300k web data
*   120k telephone conversations
'''
nlp = spacy.load('en_core_web_sm')
stopwords_set = set(stopwords.words('english'))
stopwords_set.add('amp') # ampersand error on this dataset

def getNE(tweets, maxnum = 4000):
    '''
    args:
        tweets: List of strings of tweet text
        maxnum: int, maximum number of tweets to label with entities at once (limited by memory)

    returns: dictionary mapping named entities to their counts
    '''
    items = collections.defaultdict(int)
    text = []
    k = 0
    for i,tweet in enumerate(tweets):
        if k == maxnum or i == len(tweets) - 1:
            text = ' '.join(text)
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label not in [391,396,397,384]: # labels for non-named entities (remove places too)
                    items[ent.text] += 1
            k = 0
            text = []
        else:
            k = k + 1
            text.append(tweet)
    return items

def preprocess(strings):
    clean_text = []
    for text in strings:
        tokens = word_tokenize(text)
        tokens = map(lambda s: s.lower(), tokens)
        tokens = filter(lambda tok: tok.isalnum() and (tok not in stopwords_set) and (tok != 'https') and ('//t' not in tok) and ('covid' not in tok) and ('corona' not in tok) and (len(tok) > 2), tokens)
        clean_text.append(list(tokens))
    return clean_text

def preprocessString(strings):
    word_string = ' '.join(strings)
    tokens = word_tokenize(word_string)
    tokens = map(lambda s: s.lower(), tokens)
    tokens = filter(lambda tok: tok.isalnum() and (tok not in stopwords_set) and (tok != 'https') and ('//t' not in tok) and ('covid' not in tok) and ('corona' not in tok) and (len(tok) > 3), tokens)
    clean_text = list(tokens)
    return clean_text

def lemmatize(strings):
    lemmatizer = WordNetLemmatizer()
    tagmap = {'NOUN':NOUN, 'VERB':VERB, 'ADJ':ADJ, 'ADV':ADV}
    tweetset_cleaned = []
    for tweet in strings:
        tokens = word_tokenize(tweet)
        tokens = map(lambda s: s.lower(), tokens)
        tokens = filter(lambda tok: tok.isalnum() and (tok not in stopwords_set) and (tok != 'https') and ('//t' not in tok) and ('covid' not in tok) and ('corona' not in tok) and (len(tok) > 3), tokens)
        text = list(tokens)
        tagged = pos_tag(text, tagset='universal')
        cleaned_tweet = []
        for i, (word,tag) in enumerate(tagged):
            cleaned_tweet.append(lemmatizer.lemmatize(word, pos=tagmap.get(tag,NOUN)))
        tweetset_cleaned.append(cleaned_tweet)
    return tweetset_cleaned

def buildDocuments(ents, tweets):
    '''
    args:
        ents: list of tuples (entity, entitycount)
        tweets: list of strings
    returns:
        list of documents
    '''
    documents = []
    for i,ent in enumerate(ents):
        documents.append([])
        for j,tweet in enumerate(tweets):
            if ent in tweet.lower():
                documents[i].append(tweet)
    return documents


def get_top_NEs(tweets, num=100, verbose=True):
    '''
    Gets the top {num} named entities from {data}. Requires data.text to exist.
    '''
    NES = getNE(tweets)
    most = Counter(NES).most_common(num)
    if verbose:
        print(f"Top {num} named entities:")
        print(most)
    return most

num_nes = 300

cleaned = preprocess(data['text'])
tweets = [' '.join(item) for item in cleaned]

# load NEs if available
if os.path.isfile('vars/{}_seed_{}_tweets_top_{}_nes.pkl'.format(args.seed, per_file,num_nes)):
    most = pickle.load(open('vars/{}_seed_{}_tweets_top_{}_nes.pkl'.format(args.seed, per_file,num_nes), "rb"))
else:
    most = get_top_NEs(tweets,num=num_nes)

    pickle.dump(most, open('vars/{}_seed_{}_tweets_top_{}_nes.pkl'.format(args.seed, per_file,num_nes), "wb"))

groupings = [item[0] for item in most]

'''Making array of strings to store the 100 documents corresponding to the 100 most common named entities.

Note: If tweet does not belong to any grouping (has label -1) it is excluded. Not sure if this is a good assumption. Could potentially assign these tweets to a random group or create random groups for the outliers.
Choose number of most common NEs in order to minimize outliers that are groupless.
'''

clean_text = []
list(map(clean_text.extend, cleaned))
cleaned = None
num_best_collocations = 1000

if os.path.isfile('vars/{}_seed_{}_tweets_{}_collocations.pkl'.format(args.seed, per_file,num_best_collocations)):
    bigrams = pickle.load(open('vars/{}_seed_{}_tweets_{}_collocations.pkl'.format(args.seed, per_file,num_best_collocations), "rb"))
else:
    bigram_collocation = BigramCollocationFinder.from_words(clean_text)
    bigrams = bigram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, num_best_collocations)
    pickle.dump(bigrams, open('vars/{}_seed_{}_tweets_{}_collocations.pkl'.format(args.seed, per_file,num_best_collocations), "wb"))

print(f"{num_best_collocations} best bigrams:")
print(bigrams)

for bigram in bigrams:
  groupings.append(' '.join(bigram))

# eliminate duplicates
groupings = list(set(groupings))
clean_text = None
documents = buildDocuments(groupings, tweets)
strings = [' '.join(doc) for doc in documents]
"""### LDA on documents of grouped tweets"""

word_string = ""
clean_text = lemmatize(strings)
dictionary_LDA = corpora.Dictionary(clean_text)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in clean_text]
corpus = [item for item in corpus if item != []]
num_topics = 8
lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                  id2word=dictionary_LDA, \
                                  passes=4, alpha=[.01]*num_topics, \
                                  eta=[.01]*len(dictionary_LDA.keys()))

for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=15):
    print(str(i)+": "+ topic)
    print()

strings = None
clean_text = None
"""
Assign topic to each document
"""

topic_assignments = []
for k in range(len(corpus)):
  prediction = lda_model[corpus[k]]
  topic_assignments.append(prediction[0][0])
  best = prediction[0][1]
  for topic in prediction:
    if topic[1] > best:
      best = topic[1]
      topic_assignments[k] = topic[0]

print(topic_assignments)

tweets = []
dates = []
sentiment = []

sen = SentimentIntensityAnalyzer()
for index, row in data.iterrows():
    tweets.append(row['text'])
    dates.append(int(row.created_at.dayofyear))
    sentiment.append(sen.polarity_scores(row['text'])['compound'])


print("Test tweet " + str(tweets[0]))
toks = preprocess(tweets)

topicmap = [collections.defaultdict(int) for i in range(num_topics)]
sentiment_map = [collections.defaultdict(int) for i in range(num_topics)]
topicstrings = collections.defaultdict(list)

# Predict topic and sentiment
for i,tokset in enumerate(toks):
    pred = lda_model[dictionary_LDA.doc2bow(tokset)]
    cdict = topicmap[pred[0][0]]
    sentiment_dict = sentiment_map[pred[0][0]]
    cdict[dates[i]] += 1
    topicstrings[pred[0][0]].append(' '.join(tokset))
    if sentiment[i] >= .05:
        sentiment_dict[dates[i]] += 1
    elif sentiment[i] <= -.05:
        sentiment_dict[dates[i]] -= 1


pickle.dump(topicmap, open('vars/{}_seed_{}_tweets_top_{}_topicmap.pkl'.format(args.seed,per_file,num_nes), "wb"))
pickle.dump(sentiment_map, open('vars/{}_seed_{}_tweets_top_{}_sentmap.pkl'.format(args.seed, per_file, num_nes), "wb"))

# Collect topic and sentiment predictions
for i,topic in enumerate(topicmap):
    print(f"Topic {i}:")
    counts = []
    sents = []
    for j in range(89,106): # days of year we're looking at!
        counts.append(topic[j])
        if counts[-1] == 0:
            sc = 0
        else:
            sc = sentiment_map[i][j]/counts[-1]
        sents.append(sc)
    print(counts)
    print([round(sent,2) for sent in sents])


pickle.dump(counts, open('vars/{}_seed_{}_tweets_top_{}_counts.pkl'.format(args.seed, per_file,num_nes), "wb"))
pickle.dump(sents, open('vars/{}_seed_{}_tweets_top_{}_sents.pkl'.format(args.seed, per_file,num_nes), "wb"))

"""Generate word cloud for each topic"""


from wordcloud import WordCloud as wc
import matplotlib.pyplot as plt

def show_wordcloud(word_string, filepath):
    # Create and generate a word cloud image:
    wordcloud = wc().generate(word_string)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filepath)

for key in topicstrings.keys():
    print(f"Building wordcloud for topic {key}.")
    show_wordcloud(' '.join(topicstrings[key]), '{}_seed_{}_topic_{}_wordcloud.png'.format(args.seed, per_file, key))

pickle.dump(topicstrings, open('vars/{}_seed_{}_tweets_top_{}_topicstrings.pkl'.format(args.seed, per_file,num_nes), "wb"))
