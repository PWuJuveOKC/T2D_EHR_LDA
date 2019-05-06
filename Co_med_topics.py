from __future__ import division, print_function
import pandas as pd
import pickle as pk
import gensim
import re
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis.gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

domain = 'co_medication'
RX1 = pd.read_csv('~/Desktop/HC Analysis/RX_processed/RX1.csv')

with open("Datasets/ids_subject1.pkl", "rb") as fp:
    ids_group = pk.load(fp)


RX2 = RX1[RX1['patient_id'].isin(ids_group['Metformin_Insulin']) |
          RX1['patient_id'].isin(ids_group['Metformin_Glyburide']) |
          RX1['patient_id'].isin(ids_group['Metformin_Glipizide'])].copy()

RX3 = RX2[RX2.generic_name.apply(lambda x: ('METFORMIN' not in str(x).upper()) &
                                           ('INSULIN' not in str(x).upper()) & ('GLYBURIDE' not in str(x).upper()) &
                                           ('GLIPIZIDE' not in str(x).upper())) == True].copy()

RX3['drug_name_lower'] = RX3.generic_name.apply(lambda x: str(x).lower())

# merge with lab data to get second line time
lab = pd.read_csv('Datasets_Lab/' + 'HbA1c' + '_time.csv')
lab = lab.drop_duplicates(['patient_id', 'second_time'])
RX3 = pd.merge(RX3, lab, on='patient_id', how='inner')

# prior to second line
RX4 = RX3[RX3['start_date'] <= RX3['second_time']].copy()
df = RX4[['drug_name_lower', 'patient_id']].groupby(['patient_id'])['drug_name_lower']\
    .apply(lambda x: ' '.join(x)).reset_index()

# Trigrams Topic Model
# Convert to list
data = df['drug_name_lower'].values.tolist()
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]


def sent_to_words(sentences):
    for sent in sentences:
        yield(gensim.utils.simple_preprocess(str(sent), deacc=True))

data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=5)
trigram = gensim.models.Phrases(bigram[data_words], threshold=5)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Form Trigrams
processed_docs = make_trigrams(data_words)

res = []
for sentence in processed_docs:
    new_sent = [token for token in sentence if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3]
    res.append(new_sent)

# Bag of Words in Dataset
dictionary = gensim.corpora.Dictionary(res)
dictionary.filter_extremes(no_below=5, no_above=1, keep_n=10000)
bow_corpus = [dictionary.doc2bow(doc) for doc in res]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# LDA using bag of words
num_topic = 5
FIXED_SEED = 11
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       iterations=500, random_state=FIXED_SEED)

for idx, topic in lda_model.print_topics(-1, 50):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# topics embedding
doc_topics = []
for bow in bow_corpus:
    doc_topics.append(dict(lda_model.get_document_topics(bow,  per_word_topics=False)))

topics_emb = pd.DataFrame(doc_topics)
topics_emb.fillna(0, inplace=True)
topics_emb.columns = [domain + '_' + 'topic_' + str(i) for i in range(1, num_topic + 1)]
topics_emb['patient_id'] = df.patient_id
topics_emb.to_csv('Datasets/{}_topic_model_feat.csv'.format(domain), index=None)

# for each document
for index, score in sorted(lda_model[bow_corpus[5]], key=lambda tup: -1 * tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 5)))

# visualization
lda_display_bow = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_display_bow, 'Output/HTML/lda_bow_co_med.html')

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=res, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
