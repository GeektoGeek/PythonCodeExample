
import tensorflow as ts
import pandas as pd
import os
import seaborn as sns
import numpy as np
import pandasql as ps
import pandas as pd
import sqlite3
import pandas.io.sql as psql
import ast
import re
import datetime
import seaborn as sb
import sklearn
from pandasql import sqldf
from pandasql import *
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
import featuretools as ft
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ps = lambda q: sqldf(q, globals())
from scipy.stats import pearsonr
sns.set(style='white', font_scale=1.2)

sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))


a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))

pysqldf = lambda q: sqldf(q, globals())


### NPS_2019 Raw Data
SurveyResponses = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_BoardMeeting/NPSResponses01222020.csv',encoding= 'iso-8859-1')

NPS_Pool = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_BoardMeeting/NPSPool.csv',encoding= 'iso-8859-1')

SFDCAdvisors = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_BoardMeeting/SFDCAdvisors.csv',encoding= 'iso-8859-1')

SurveyResponses.info()


def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

SurveyResponses1= SurveyResponses.iloc[:,[0,18]]

SurveyResponses2= SurveyResponses.iloc[:,[18]]

SurveyResponses2 = SurveyResponses2.replace(np.nan, '', regex=True)

SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof']=SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].astype(str)

SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'] =SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].apply(lambda x:pre_process(x))

## SurveyResponses2 = SurveyResponses2.replace(np.nan, '', regex=True)

### Bring the Sequnrtial Email Processed Data

#show the second 'text' just for fun

###Creating Vocabulary and Word Counts for IDF
## Convert the Email Subject to a list
##EmailSubjectNLP['EmailSubject'].to_list()

SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].to_list()

tfidf = TfidfVectorizer(analyzer='word', stop_words = 'english')

##score = tfidf.fit_transform(EmailSubjectNLP['EmailSubject'])

score = tfidf.fit_transform(SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'])

score1 = tfidf.fit_transform(SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'])

# New data frame containing the tfidf features and their scores
tfidf_df = pd.DataFrame(score.toarray(), columns=tfidf.get_feature_names())

tfidf_df1 = pd.DataFrame(score1.toarray(), columns=tfidf.get_feature_names())

# Filter the tokens with tfidf score greater than 0.3
tokens_above_threshold = tfidf_df.max()[tfidf_df.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1 = tfidf_df1.max()[tfidf_df1.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold2= pd.DataFrame([tokens_above_threshold1])


tokens_above_threshold2 = tokens_above_threshold2.to_csv (r'C:\Users\test\Documents\EmailEngagement\tokens_above_threshold2.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### let try to do the WordCloud

# Start with one review:
SurveyResponses2 = SurveyResponses2.replace(np.nan, '', regex=True)

text = SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'][0:142]

wordcloud2 = WordCloud().generate(' '.join(text))

# Create and generate a word cloud image:
## wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig('wordcloud2.png', facecolor='k', bbox_inches='tight')

#### LatentDirichletAllocation as LDA

SurveyResponses3=  SurveyResponses2

# Remove punctuation
SurveyResponses3['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'] = SurveyResponses3['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
SurveyResponses3['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'] = SurveyResponses3['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].map(lambda x: x.lower())
# Print out the first rows of papers
SurveyResponses3['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].head()


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(SurveyResponses3['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(SurveyResponses3['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint

%matplotlib inline

from gensim import corpora, models

text1= text.tolist()

dataset = [d.split() for d in text1]

dictionary = Dictionary(dataset)

corpus = [dictionary.doc2bow(text) for text in dataset]

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)

lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics

lsitopics = lsimodel.show_topics(formatted=False)

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

hdpmodel.show_topics()

hdptopics = hdpmodel.show_topics(formatted=False)

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

import pyLDAvis.gensim

pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

ldatopics = ldamodel.show_topics(formatted=False)

p = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(p, 'lda.html')

### Lets start with Athene data
### You need to reload the data because otherwise nan is not removed
######

########

SurveyResponses = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_BoardMeeting/NPSResponses01222020.csv',encoding= 'iso-8859-1')

SurveyResponses.info()

AtheneComment= SurveyResponses.iloc[:,[20]]

AtheneComment.info()



AtheneComment = AtheneComment.replace(np.nan, '', regex=True)

AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof']=AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof'].astype(str)

AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof'] =AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof'].apply(lambda x:pre_process(x))

AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof'].to_list()

tfidf_Athene = TfidfVectorizer(analyzer='word', stop_words = 'english')

##score = tfidf.fit_transform(EmailSubjectNLP['EmailSubject'])

score__Athene = tfidf_Athene.fit_transform(AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof'])

score1__Athene = tfidf_Athene.fit_transform(AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof'])

# New data frame containing the tfidf features and their scores
tfidf_df_Athene = pd.DataFrame(score__Athene.toarray(), columns=tfidf_Athene.get_feature_names())

tfidf_df1_Athene = pd.DataFrame(score1__Athene.toarray(), columns=tfidf_Athene.get_feature_names())

# Filter the tokens with tfidf score greater than 0.3
tokens_above_threshold_Athene = tfidf_df_Athene.max()[tfidf_df_Athene.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1_Athene = tfidf_df_Athene.max()[tfidf_df1_Athene.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1_Athene= pd.DataFrame([tokens_above_threshold1_Athene])



tokens_above_threshold1_Athene = tokens_above_threshold1_Athene.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_BoardMeeting\tokens_above_threshold1_Athene.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

# Start with one review:
##tokens_above_threshold1_Athene = tokens_above_threshold1_Athene.replace(np.nan, '', regex=True)

text1 = AtheneComment['WhydidyougiveAtheneBCA2.0FIAascoreof'][0:142]

wordcloud3 = WordCloud().generate(' '.join(text1))

# Create and generate a word cloud image:
## wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig('wordcloud3.png', facecolor='k', bbox_inches='tight')

#### LatentDirichletAllocation as LDA

AtheneComment1 =  AtheneComment

# Remove punctuation
AtheneComment1['WhydidyougiveAtheneBCA2.0FIAascoreof'] = AtheneComment1['WhydidyougiveAtheneBCA2.0FIAascoreof'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
AtheneComment1['WhydidyougiveAtheneBCA2.0FIAascoreof'] = AtheneComment1['WhydidyougiveAtheneBCA2.0FIAascoreof'].map(lambda x: x.lower())
# Print out the first rows of papers
AtheneComment1['WhydidyougiveAtheneBCA2.0FIAascoreof'].head()


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string1 = ','.join(list(AtheneComment1['WhydidyougiveAtheneBCA2.0FIAascoreof'].values))
# Create a WordCloud object
wordcloud5 = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud5.generate(long_string1)
# Visualize the word cloud
wordcloud5.to_image()

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# Helper function
def plot_10_most_common_words1(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(AtheneComment1['WhydidyougiveAtheneBCA2.0FIAascoreof'])
# Visualise the 10 most common words
plot_10_most_common_words1(count_data, count_vectorizer)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics_Athene(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda_Athene = LDA(n_components=number_topics, n_jobs=-1)
lda_Athene.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda_Athene, count_vectorizer, number_words)

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint

%matplotlib inline

from gensim import corpora, models

text1= text.tolist()

dataset_athene = [d.split() for d in text1]

dictionary_athene = Dictionary(dataset_athene)

corpus_athene = [dictionary.doc2bow(text) for text in dataset_athene]

lsimodel_athene = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary_athene)

lsimodel_athene.show_topics(num_topics=5)  # Showing only the top 5 topics

lsitopics_athene = lsimodel.show_topics(formatted=False)

hdpmodel_athene = HdpModel(corpus=corpus, id2word=dictionary)

hdpmodel_athene.show_topics()

hdptopics_athene = hdpmodel.show_topics(formatted=False)

ldamodel_athene = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

import pyLDAvis.gensim

pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(ldamodel_athene, corpus_athene, dictionary_athene)

ldatopics_athene = ldamodel.show_topics(formatted=False)

p = pyLDAvis.gensim.prepare(ldamodel_athene, corpus_athene, dictionary_athene)
pyLDAvis.save_html(p, 'lda_athene.html')


### Lets start with Trans America data

SurveyResponses = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_BoardMeeting/NPSResponses01222020.csv',encoding= 'iso-8859-1')

SurveyResponses.info()
TransAmericaComment= SurveyResponses.iloc[:,[22]]

TransAmericaComment = TransaAmericaComment.replace(np.nan, '', regex=True)

TransAmericaComment.info()

TransAmericaComment['WhydidyougiveTAFIAscoreof']= TransaAmericaComment['WhydidyougiveTAFIAscoreof'].astype(str)

TransAmericaComment['WhydidyougiveTAFIAscoreof'] =TransaAmericaComment['WhydidyougiveTAFIAscoreof'].apply(lambda x:pre_process(x))

TransAmericaComment['WhydidyougiveTAFIAscoreof'].to_list()

tfidf_TransAmerica = TfidfVectorizer(analyzer='word', stop_words = 'english')

##score = tfidf.fit_transform(EmailSubjectNLP['EmailSubject'])

score__TransAmerica = tfidf_TransaAmerica.fit_transform(TransaAmericaComment['WhydidyougiveTAFIAscoreof'])

score1__TransAmerica = tfidf_TransaAmerica.fit_transform(TransaAmericaComment['WhydidyougiveTAFIAscoreof'])

# New data frame containing the tfidf features and their scores
tfidf_df_TransAmerica = pd.DataFrame(score__TransaAmerica.toarray(), columns=tfidf_TransaAmerica.get_feature_names())

tfidf_df1_TransAmerica = pd.DataFrame(score1__TransaAmerica.toarray(), columns=tfidf_TransaAmerica.get_feature_names())

# Filter the tokens with tfidf score greater than 0.3
tokens_above_threshold_TransAmerica = tfidf_df_TransAmerica.max()[tfidf_df_TransAmerica.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1_TransAmerica = tfidf_df_TransAmerica.max()[tfidf_df_TransAmerica .max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1_TransAmerica= pd.DataFrame([tokens_above_threshold1_TransAmerica])


tokens_above_threshold1_TransAmerica = tokens_above_threshold1_TransAmerica.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_BoardMeeting\tokens_above_threshold1_TransAmerica.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


# Start with one review:


text2 = TransAmericaComment['WhydidyougiveTAFIAscoreof'][0:142]

wordcloud4 = WordCloud().generate(' '.join(text2))

# Create and generate a word cloud image:
## wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud4, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig('wordcloud4.png', facecolor='k', bbox_inches='tight')

#### LatentDirichletAllocation as LDA

# Remove punctuation

TransAmericaComment1 =  TransAmericaComment

TransAmericaComment1['WhydidyougiveTAFIAscoreof'] = TransAmericaComment['WhydidyougiveTAFIAscoreof'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
TransAmericaComment1['WhydidyougiveTAFIAscoreof'] = TransAmericaComment1['WhydidyougiveTAFIAscoreof'].map(lambda x: x.lower())
# Print out the first rows of papers
TransAmericaComment1['WhydidyougiveTAFIAscoreof'].head()

# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string2 = ','.join(list(TransAmericaComment1['WhydidyougiveTAFIAscoreof'].values))
# Create a WordCloud object
wordcloud_TA = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud_TA.generate(long_string2)
# Visualize the word cloud
wordcloud_TA.to_image()

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# Helper function
def plot_10_most_common_words2(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
    
    # Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(TransAmericaComment1['WhydidyougiveTAFIAscoreof'])
# Visualise the 10 most common words
plot_10_most_common_words2(count_data, count_vectorizer)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics_TA(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda_TA = LDA(n_components=number_topics, n_jobs=-1)
lda_TA.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda_TA, count_vectorizer, number_words)

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint

%matplotlib inline

from gensim import corpora, models

text2= text.tolist()

dataset_TA = [d.split() for d in text2]

dictionary_TA = Dictionary(dataset_TA)

corpus_TA = [dictionary.doc2bow(text) for text in dataset_TA]

lsimodel_TA = LsiModel(corpus=corpus_TA, num_topics=10, id2word=dictionary_athene)

lsimodel_TA.show_topics(num_topics=5)  # Showing only the top 5 topics

lsitopics_TA = lsimodel.show_topics(formatted=False)

hdpmodel_TA = HdpModel(corpus=corpus, id2word=dictionary)

hdpmodel_TA.show_topics()

hdptopics_TA = hdpmodel.show_topics(formatted=False)

ldamodel_TA = LdaModel(corpus=corpus_TA, num_topics=10, id2word=dictionary)

import pyLDAvis.gensim

pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(ldamodel_TA, corpus_athene, dictionary_athene)

ldatopics_TA = ldamodel.show_topics(formatted=False)

p = pyLDAvis.gensim.prepare(ldamodel_TA, corpus_TA, dictionary_athene)
pyLDAvis.save_html(p, 'lda_TA.html')


### State of Isthereanythingelseyouwouldliketoshare

SurveyResponses = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_BoardMeeting/NPSResponses01222020.csv',encoding= 'iso-8859-1')

SurveyResponses.info()

IsthereAnything= SurveyResponses.iloc[:,[23]]

IsthereAnything.info() 

IsthereAnythingComment = IsthereAnything.replace(np.nan, '', regex=True)

IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare']=IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'].astype(str)

IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'] =IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'].apply(lambda x:pre_process(x))

IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'].to_list()

tfidf_IsthereAnything = TfidfVectorizer(analyzer='word', stop_words = 'english')

##score = tfidf.fit_transform(EmailSubjectNLP['EmailSubject'])

score__IsthereAnything = tfidf_IsthereAnything.fit_transform(IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'])

score1__IsthereAnything = tfidf_IsthereAnything.fit_transform(IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'])

# New data frame containing the tfidf features and their scores
tfidf_df_IsthereAnything = pd.DataFrame(score__IsthereAnything.toarray(), columns=tfidf_IsthereAnything.get_feature_names())

tfidf_df1_IsthereAnything = pd.DataFrame(score1__IsthereAnything.toarray(), columns=tfidf_IsthereAnything.get_feature_names())

# Filter the tokens with tfidf score greater than 0.3
tokens_above_threshold_IsthereAnything = tfidf_df_IsthereAnything.max()[tfidf_df_IsthereAnything.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1_IsthereAnything = tfidf_df_IsthereAnything.max()[tfidf_df_IsthereAnything.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1_IsthereAnything= pd.DataFrame([tokens_above_threshold1_IsthereAnything])

tokens_above_threshold1_IsthereAnything = tokens_above_threshold1_IsthereAnything.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_BoardMeeting\tokens_above_threshold1_IsthereAnything.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

# Start with one review:


text3 = IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'][0:142]

wordcloud5 = WordCloud().generate(' '.join(text3))

# Create and generate a word cloud image:
## wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud5, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig('wordcloud5.png', facecolor='k', bbox_inches='tight')

#### LatentDirichletAllocation as LDA

# Remove punctuation

IsthereAnythingComment1 =  IsthereAnythingComment

IsthereAnythingComment1['Isthereanythingelseyouwouldliketoshare'] = IsthereAnythingComment['Isthereanythingelseyouwouldliketoshare'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
IsthereAnythingComment1['Isthereanythingelseyouwouldliketoshare'] = IsthereAnythingComment1['Isthereanythingelseyouwouldliketoshare'].map(lambda x: x.lower())
# Print out the first rows of papers
IsthereAnythingComment1['Isthereanythingelseyouwouldliketoshare'].head()

# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string3 = ','.join(list(IsthereAnythingComment1['Isthereanythingelseyouwouldliketoshare'].values))
# Create a WordCloud object
wordcloud_IsthereAny = WordCloud(background_color="white", max_words=10000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud_IsthereAny.generate(long_string2)
# Visualize the word cloud
wordcloud_IsthereAny.to_image()

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# Helper function
def plot_10_most_common_words3(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
            
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
    
### Initialise the count vectorizer with the English stop words
    
### Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(IsthereAnythingComment1['Isthereanythingelseyouwouldliketoshare'])

# Visualise the 10 most common words
plot_10_most_common_words2(count_data, count_vectorizer)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics_IsAnything(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below to get any number of combinations
### This is where we need to change a lot of stuff    
    
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda_IsAny = LDA(n_components=number_topics, n_jobs=-1)
lda_IsAny.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda_IsAny, count_vectorizer, number_words)

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint

%matplotlib inline

from gensim import corpora, models

text4= text.tolist()

dataset_IsAny = [d.split() for d in text4]

dictionary_IsAny = Dictionary(dataset_IsAny)

corpus_IsAny = [dictionary.doc2bow(text4) for text in dataset_IsAny]

lsimodel_IsAny = LsiModel(corpus=corpus_IsAny, num_topics=10, id2word=dictionary_athene)

lsimodel_IsAny.show_topics(num_topics=5)  # Showing only the top 5 topics

lsitopics_IsAny = lsimodel.show_topics(formatted=False)

hdpmodel_IsAny = HdpModel(corpus=corpus, id2word=dictionary)

hdpmodel_IsAny.show_topics()

hdptopics_IsAny = hdpmodel.show_topics(formatted=False)

ldamodel_IsAny = LdaModel(corpus=corpus_IsAny, num_topics=10, id2word=dictionary)

import pyLDAvis.gensim

pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(ldamodel_IsAny, corpus_IsAny, dictionary_IsAny)

ldatopics_IsAny = ldamodel.show_topics(formatted=False)

p = pyLDAvis.gensim.prepare(ldamodel_IsAny, corpus_IsAny, dictionary_IsAny)

pyLDAvis.save_html(p, 'ldamodel_IsAny.html')































