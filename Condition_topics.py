from __future__ import division, print_function
import pandas as pd
import pickle as pk
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
from gensim.models import CoherenceModel
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

domain = 'condition'
code = 'ICD'
col_names = ['patient_id', 'account', 'diagnosis_time', 'diagnosis_type', 'diagnosis_code', 'description']

condition = pd.read_csv('~/Desktop/HC Analysis/New Data/diagnosis.csv', names=col_names)

with open("Datasets/ids_subject1.pkl", "rb") as fp:
    ids_group = pk.load(fp)


condition2 = condition[condition['patient_id'].isin(ids_group['Metformin_Insulin']) |
                       condition['patient_id'].isin(ids_group['Metformin_Glyburide']) |
                       condition['patient_id'].isin(ids_group['Metformin_Glipizide'])].copy()

# remove Type II diabetes diag code
remove = {'250.00', '250.02', '250.10', '250.12', '250.20', '250.22', '250.30', '250.32', '250.40', '250.40',
          '250.42', '250.50', '250.52', '250.60', '250.62', '250.70', '250.72', '250.80', '250.82', '250.90', '250.92'}
condition2 = condition2[~condition2.diagnosis_code.isnull()]
condition2 = condition2[~condition2.diagnosis_code.isin(remove)]
condition2['ICD'] = condition2['diagnosis_code'].apply(lambda x: x.strip())

# merge with lab data to get second line time
lab = pd.read_csv('Datasets_Lab/' + 'HbA1c' + '_time.csv')
lab = lab.drop_duplicates(['patient_id', 'second_time'])
condition3 = pd.merge(condition2, lab, on='patient_id', how='inner')

# prior to second line
condition4 = condition3[condition3['diagnosis_time'] <= condition3['second_time']].copy()

# output to condition-description file
condition5 = condition4.drop_duplicates(subset='ICD', keep='first').copy()
condition5 = condition5[['ICD', 'description']].copy()
condition5['description'] = condition5.description.apply(lambda x: str(x).lower())
condition5.to_csv('Datasets/ICD_description.csv', index=None)

df = condition4[[code, 'patient_id']].groupby(['patient_id'])[code].apply(lambda x: ' '.join(x)).reset_index()

# Convert to list
data = df[code].values.tolist()
res = [sent.split(' ') for sent in data]

# Bag of Words
dictionary = gensim.corpora.Dictionary(res)
count = 0

dictionary.filter_extremes(no_below=5, no_above=1.0, keep_n=10000)
bow_corpus = [dictionary.doc2bow(doc) for doc in res]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# LDA using BoW
num_topic = 5
seed = 11
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       iterations=500, random_state=seed)

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
lda_display_bow = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics=True)
pyLDAvis.save_html(lda_display_bow, 'Output/HTML/lda_bow_' + domain + '.html')

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=res, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)