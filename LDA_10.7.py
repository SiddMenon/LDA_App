
# coding: utf-8

# # LDA demo
# ### source: https://gist.github.com/georgehc/d2353feef7e09b4b53fc087d44f75954

# In[72]:


import nltk
import json
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random


# In[2]:


content = []
for line in open('JACS.json', 'r'):
    content.append(json.loads(line))


# ## To load eupmc.json

# In[39]:


content = []
with open('eupmc.json') as json_data:
    content = json.load(json_data)


# In[3]:


Total = []
for c in content:
    ##using both title and content
    total = c['Title']
    Total.append(total)
    
#print(Total)


# In[4]:


vocab_size = 1000
from sklearn.feature_extraction.text import CountVectorizer

# document frequency (df) means number of documents a word appears in
tf_vectorizer = CountVectorizer(max_df=0.95,
                                min_df=2,
                                max_features=vocab_size,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(Total)


# In[5]:


num_topics = 20

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', random_state=95865)
lda.fit(tf)


# In[6]:


import numpy as np
topic_word_distributions = np.array([topic_word_pseudocounts / np.sum(topic_word_pseudocounts)
                                     for topic_word_pseudocounts in lda.components_])


# In[7]:


num_top_words = 10

print('Displaying the top %d words per topic and their probabilities within the topic...' % num_top_words)
print()

import numpy as np
for topic_idx in range(num_topics):
    print('[Topic %d]' % topic_idx)
    sort_indices = np.argsort(topic_word_distributions[topic_idx])[::-1]
    for rank in range(num_top_words):
        word_idx = sort_indices[rank]
        print('%s: %f' % (tf_vectorizer.get_feature_names()[word_idx], topic_word_distributions[topic_idx, word_idx]))
    print()


# # Another Way of doing LDA
# ### Source: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

# In[8]:


from nltk.corpus import stopwords 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(entry).split() for entry in Total]

#print(doc_clean)


# In[79]:


import gensim
from gensim import corpora,models

random.shuffle(Total)

training = Total[:round(len(Total)*0.6)]
test = Total[round(len(Total)*0.6):]

doc_clean_train = [clean(entry).split() for entry in training]
doc_clean_test = [clean(entry).split() for entry in test]
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary_tr = corpora.Dictionary(doc_clean_train)
dictionary_te = corpora.Dictionary(doc_clean_test)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix_te = [dictionary_te.doc2bow(doc) for doc in doc_clean_test]


#mystring = mystring..decode(‘utf-8’)


# In[83]:


tfidf = models.TfidfModel(doc_term_matrix_tr)
corpus_tfidf = tfidf[doc_term_matrix_tr]
corpus_tfidf_te = tfidf[doc_term_matrix_te]


# In[81]:


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)


# In[82]:


Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix_tr, num_topics=5, id2word = dictionary_tr, passes=50)


# In[46]:


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx+1, topic))


# In[84]:


lda_model_tfidf.log_perplexity(corpus_tfidf_te)


# In[85]:


ldamodel.log_perplexity(doc_term_matrix_te)


# In[59]:


#Dynamic visualization of model results 
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tfidf, dictionary)
vis


# In[93]:


##pick optimal number of topics 
s = []
for i in range(1,40):
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=i, id2word=dictionary_tr, passes=2, workers=4)
    score = lda_model_tfidf.log_perplexity(doc_term_matrix_te)
    s.append(score)


# In[94]:


#If the coherence score seems to keep increasing, it may make better sense to pick the model 
#that gave the lowest perplexity before flattening out. This is exactly the case here.
x = range(1,40)
plt.plot(x, s)
plt.xlabel("Num Topics")
plt.ylabel("Perplexity")
plt.legend(("Perplexity"), loc='best')
plt.show()
#Looks like 34 is optimal here? 

