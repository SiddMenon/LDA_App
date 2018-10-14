
# coding: utf-8

# # Author Topic Modeling
# ### Source: https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb

# ## 1. Processing Text, Vectorize Author List

# In[195]:


import nltk
import json
content = []

content = []
with open('total.json') as json_data:
    content = json.load(json_data)

#for line in open('JACS.json', 'r'):
#    content.append(json.loads(line))
    
print(len(content))


# In[196]:


# Get all author names and their corresponding document IDs.
author2doc = dict()

i = 0
for entry in content:
    sender = entry['Sender'].replace('\n',' ')
    if not author2doc.get(sender):
        # This is a new author.
        #author2doc[sender] = []
        author2doc[sender] = [i]
    # Add document IDs to author.
    else:
        author2doc[sender].append(i)
    i = i + 1
    
i = 0    
for entry in content:
    receiver = entry['Receiver'].replace('\n',' ')
    if not author2doc.get(receiver):
        # This is a new author.
        author2doc[receiver] = []
        author2doc[receiver] = [i]
    # Add document IDs to author.
    else:
        author2doc[receiver].append(i)
    i = i + 1
    
    
#print(author2doc)


# In[197]:


import spacy
nlp = spacy.load('en')


# In[198]:


abstract = []
for entry in content:
    title = entry['Title'].replace('\n',' ')
    #sender = entry['Sender'].replace('\n',' ')
    #receiver = entry['Receiver'].replace('\n',' ')
    abst = entry['Content'].replace('\n',' ')
    entry_str = title+' '+abst
    abstract.append(entry_str)
#print(abstract)


# In[199]:


from nltk.corpus import stopwords
d = {}
stopword = stopwords.words('english')


# In[200]:


get_ipython().run_cell_magic('time', '', 'processed_docs = []    \nfor doc in nlp.pipe(abstract, n_threads=4, batch_size=100):\n    # Process document using Spacy NLP pipeline.\n    \n    ents = doc.ents  # Named entities.\n\n    # Keep only words (no numbers, no punctuation).\n    # Lemmatize tokens, remove punctuation and remove stopwords.\n    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]\n\n    # Remove common words from a stopword list.\n    doc = [token for token in doc if token not in stopword]\n\n    # Add named entities, but only if they are a compound of more than word.\n    doc.extend([str(entity) for entity in ents if len(entity) > 1])\n    \n    processed_docs.append(doc)')


# In[193]:


abstract_all = processed_docs
del processed_docs


# In[194]:


from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(abstract_all, min_count=20)
for idx in range(len(abstract_all)):
    for token in bigram[abstract_all[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            abstract_all[idx].append(token)


# In[159]:


from gensim.corpora import Dictionary
dictionary = Dictionary(abstract_all)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.3
min_wordcount = 20
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

_ = dictionary[0]  # This sort of "initializes" dictionary.id2token.


# In[160]:


corpus = [dictionary.doc2bow(doc) for doc in abstract_all]


# In[164]:


print('Number of authors: %d' % len(author2doc))
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# In[165]:


#print(len(corpus))
print(len(dictionary.id2token))


# In[166]:


#from gensim.models import AuthorTopicModel
#%time model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token, \
#                author2doc=author2doc, chunksize=2000, passes=1, eval_every=0, \
#                iterations=1, random_state=1)


# ## 2. Select Best Performing Model with highest coherence

# In[167]:


get_ipython().run_cell_magic('time', '', 'model_list = []\nfor i in range(5):\n    model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token, \\\n                    author2doc=author2doc, chunksize=2000, passes=100, gamma_threshold=1e-10, \\\n                    eval_every=0, iterations=1, random_state=i)\n    top_topics = model.top_topics(corpus)\n    tc = sum([t[1] for t in top_topics])\n    model_list.append((model, tc))')


# In[176]:


model, tc = max(model_list, key=lambda x: x[1])
print('Topic coherence: %.3e' %tc)


# In[177]:


model.save('/tmp/model.atmodel')


# In[178]:


model = AuthorTopicModel.load('/tmp/model.atmodel')


# In[179]:


model.show_topics(num_topics=10)


# In[186]:


topics = []
i = 1
for topic in model.show_topics(num_topics=10):
    words = []
    for word, prob in model.show_topic(topic[0]):
        words.append(word)
    print('Topic '+str(i)+': ')
    print(words[8]+' '+words[9])
    print(*words)
    print()
    i += 1
    topics.append(words[8]+' '+words[9])


# In[181]:


from pprint import pprint

def show_author(name):
    print('\n%s' % name)
    print('Docs:', model.author2doc[name])
    print('Topics:')
    pprint([(topics[topic[0]],topic[1]) for topic in model[name]])


# In[182]:


show_author('Jonathan L. Sessler')


# ## 3. Plotting the Authors

# In[183]:


get_ipython().run_cell_magic('time', '', 'from sklearn.manifold import TSNE\ntsne = TSNE(n_components=2, random_state=0)\nsmallest_author = 0  # Ignore authors with documents less than this.\nauthors = [model.author2id[a] for a in model.author2id.keys() if len(model.author2doc[a]) >= smallest_author]\n_ = tsne.fit_transform(model.state.gamma[authors, :])  # Result stored in tsne.embedding_')


# In[184]:


# Tell Bokeh to display plots inside the notebook.
from bokeh.io import output_notebook
output_notebook()


# In[185]:


from bokeh.models import HoverTool
from bokeh.plotting import figure, show, ColumnDataSource

x = tsne.embedding_[:, 0]
y = tsne.embedding_[:, 1]

author_names = [model.id2author[a] for a in authors]

# Radius of each point corresponds to the number of documents attributed to that author.
scale = 0.4
author_sizes = [len(model.author2doc[a]) for a in author_names]
radii = [size * scale for size in author_sizes]

source = ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            author_names=author_names,
            author_sizes=author_sizes,
            radii=radii,
        )
    )

# Add author names and sizes to mouse-over info.
hover = HoverTool(
        tooltips=[
        ("author", "@author_names"),
        ("size", "@author_sizes"),
        ]
    )

p = figure(tools=[hover, 'crosshair,pan,wheel_zoom,box_zoom,reset,save,lasso_select'])
p.scatter('x', 'y', radius='radii', source=source, fill_alpha=0.6, line_color=None)
show(p)


# ## 4. Similarity Queries

# In[170]:


from gensim.similarities import MatrixSimilarity

# Generate a similarity object for the transformed corpus.
index = MatrixSimilarity(model[list(model.id2author.values())])

# Get similarities to some author.
author_name = 'Yadong Li'
sims = index[model[author_name]]


# In[171]:


# Make a function that returns similarities based on the Hellinger distance.

from gensim import matutils
import pandas as pd

# Make a list of all the author-topic distributions.
author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

def similarity(vec1, vec2):
    '''Get similarity between two vectors'''
    dist = matutils.hellinger(matutils.sparse2full(vec1, model.num_topics),                               matutils.sparse2full(vec2, model.num_topics))
    sim = 1.0 / (1.0 + dist)
    return sim

def get_sims(vec):
    '''Get similarity of vector to all authors.'''
    sims = [similarity(vec, vec2) for vec2 in author_vecs]
    return sims

def get_table(name, top_n=10, smallest_author=1):
    '''
    Get table with similarities, author names, and author sizes.
    Return `top_n` authors as a dataframe.
    
    '''
    
    # Get similarities.
    sims = get_sims(model.get_author_topics(name))

    # Arrange author names, similarities, and author sizes in a list of tuples.
    table = []
    for elem in enumerate(sims):
        author_name = model.id2author[elem[0]]
        sim = elem[1]
        author_size = len(model.author2doc[author_name])
        if author_size >= smallest_author:
            table.append((author_name, sim, author_size))
            
    # Make dataframe and retrieve top authors.
    df = pd.DataFrame(table, columns=['Author', 'Score', 'Size'])
    df = df.sort_values('Score', ascending=False)[:top_n]
    
    return df


# In[172]:


get_table('Jonathan L. Sessler')


# In[173]:


get_table('Jonathan L. Sessler', smallest_author=3)


# In[174]:


author_dict = {}
for a in author2doc:
    topic = [(topics[t[0]]) for t in model[a]]
    author_dict[a] = topic


# In[175]:


pd.DataFrame.from_dict(author_dict, orient='index')

