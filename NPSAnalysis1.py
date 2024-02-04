
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

SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof']=SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].astype(str)

SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'] = SurveyResponses2['WhydidyougiveNationwideNHFixedIndexedAnnuityascoreof'].apply(lambda x:pre_process(x))


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