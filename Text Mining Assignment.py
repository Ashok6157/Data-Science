#!/usr/bin/env python
# coding: utf-8

# For Text Mining assignment
#  
#  ONE:
# 1) Perform sentimental analysis on the Elon-musk tweets (Exlon-musk.csv)
# 

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


musk=pd.read_csv('Elon_musk.csv',encoding='ISO-8859-1')
musk


# In[3]:


musk.drop(['Unnamed: 0'],inplace=True,axis=1)
musk


# In[4]:


musk=[x.strip() for x in musk.Text]
musk=[x for x in musk if x]
musk


# In[5]:


data=' '.join(musk)
data


# In[6]:


import string
no_punc_text=data.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[7]:


import re
no_url_text=re.sub(r'http\S+','',no_punc_text)
no_url_text


# In[8]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[9]:


from nltk.tokenize import word_tokenize
tokens=word_tokenize(no_url_text)
tokens


# In[10]:


len(tokens)


# In[11]:


from nltk.corpus import stopwords
stopword=stopwords.words('english')
stopword.append('the')
no_stop_tokens=[word for word in tokens if not word in stopword]
no_stop_tokens


# In[12]:


len(no_stop_tokens)


# In[13]:


lower_word=[x.lower() for x in no_stop_tokens]
print(lower_word)


# In[14]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_word]
stemmed_tokens


# In[15]:


len(stemmed_tokens)


# In[16]:


import spacy
lem=spacy.load('en_core_web_sm')
doc=lem(' '.join(lower_word))
doc


# In[17]:


lemma=[token.lemma_ for token in doc]
lemma


# In[18]:


clean_tweets=' '.join(lemma)
clean_tweets


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemma)


# In[20]:


vectorizer.vocabulary_


# In[21]:


vectorizer.get_feature_names()


# In[22]:


X.toarray()


# In[23]:


X.toarray().shape


# In[24]:


cv_ngram_range = CountVectorizer(analyzer = 'word', ngram_range=(1,3),max_features= 100)
bow_matrix_ngram = cv_ngram_range.fit_transform(lemma)


# In[25]:


print(cv_ngram_range.get_feature_names())
bow_matrix_ngram.toarray()


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matrix_ngram=tfidfv_ngram_max_features.fit_transform(lemma)


# In[27]:


print(tfidfv_ngram_max_features.get_feature_names())
tfidf_matrix_ngram.toarray()


# In[28]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
def plot_cloud(wordcloud):
    plt.figure(figsize=(50,40))
    plt.imshow(wordcloud)
    plt.axis('off')

stopword.append('will')
wordcloud = WordCloud(width = 5000, height = 4000, background_color = 'black', max_words=100, colormap = 'Set2', stopwords=stopword).generate(clean_tweets)

plot_cloud(wordcloud)


# In[29]:


nlp = spacy.load('en_core_web_sm')
one_block = clean_tweets
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter = True)


# In[30]:


for token in doc_block:
    print(token, token.pos_)


# In[31]:


nouns_verbs = [token.text for token in doc_block if token.pos_ in ('NOUN', 'VERB')]
print(nouns_verbs)


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(nouns_verbs)
sum_words = X.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse =True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word','count']
wf_df


# In[33]:


wf_df[0:10].plot.bar(x='word', figsize=(15,10), title = 'Top verbs and nouns')


# In[34]:


from nltk import tokenize
sentence = tokenize.sent_tokenize(' '.join(musk))
sentence


# In[35]:


sent_df = pd.DataFrame(sentence, columns = ['sentence'])
sent_df


# In[36]:


afinn = pd.read_csv('Afinn.csv', sep = ',', encoding = 'latin-1')
afinn.shape


# In[37]:


afinn.head()


# In[38]:


from matplotlib.pyplot import imread
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[40]:


sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[41]:


calculate_sentiment(text = 'amazing')


# In[42]:


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# In[43]:


sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df['word_count'].head(10)


# In[44]:


sent_df.sort_values(by='sentiment_value').tail(10)


# In[45]:


sent_df['sentiment_value'].describe()


# In[46]:


sent_df[sent_df['sentiment_value']>=15].head()


# In[47]:


sent_df['index'] = range(0,len(sent_df))


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(sent_df['sentiment_value'])


# In[49]:


plt.figure(figsize = (15,10))
sns.lineplot(y='sentiment_value', x='index', data =sent_df)


# In[50]:


sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')


# In[ ]:





# TWO:
# 1) Extract reviews of any product from ecommerce website like amazon
# 2) Perform emotion mining

# In[77]:


review = pd.read_csv('amazon.csv')
review 


# In[79]:


review = review[['verified_reviews']]
review


# In[80]:


reviews = []
for x in review.verified_reviews:
    reviews.append(x.strip())
reviews


# In[81]:


data = ' '.join(reviews)
data


# In[82]:


import string
no_punc_text = data.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[83]:


import re
no_url_text = re.sub(r'http\S+','',no_punc_text)
no_url_text


# In[84]:


no_url_text = no_url_text.lower()
no_url_text


# In[85]:


from nltk.tokenize import word_tokenize
tokens = word_tokenize(no_url_text)
tokens


# In[86]:


len(tokens)


# In[87]:


from nltk.corpus import stopwords
stopword = stopwords.words('english')
no_stop_tokens = [word for word in tokens if not word in stopword]
no_stop_tokens


# In[88]:


len(no_stop_tokens)


# In[89]:


from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in no_stop_tokens]
stemmed_tokens


# In[90]:


len(stemmed_tokens)


# In[91]:


import spacy
lem = spacy.load('en_core_web_sm')
doc = lem(' '.join(no_stop_tokens))
doc


# In[92]:


lemma = [token.lemma_ for token in doc]
lemma


# In[93]:


len(lemma)


# In[94]:


clean_review = ' '.join(lemma)


# In[95]:


clean_review


# In[96]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x= cv.fit_transform(lemma)


# In[97]:


cv.vocabulary_


# In[98]:


cv.get_feature_names()


# In[99]:


x.toarray()


# In[100]:


cv_ngram_range = CountVectorizer(analyzer = 'word', ngram_range=(1,3),max_features= 100)
bow_matrix_ngram = cv_ngram_range.fit_transform(lemma)


# In[101]:


print(cv_ngram_range.get_feature_names())
bow_matrix_ngram.toarray()


# In[102]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matrix_ngram=tfidfv_ngram_max_features.fit_transform(lemma)


# In[103]:


print(tfidfv_ngram_max_features.get_feature_names())
tfidf_matrix_ngram.toarray()


# In[104]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
def plot_cloud(wordcloud):
    plt.figure(figsize=(50,40))
    plt.imshow(wordcloud)
    plt.axis('off')

stopword.append('will')
wordcloud = WordCloud(width = 5000, height = 4000, background_color = 'black', max_words=100, colormap = 'Set2', stopwords=stopword).generate(clean_review)

plot_cloud(wordcloud)


# In[105]:


nlp = spacy.load('en_core_web_sm')
one_block = clean_review
doc_block = nlp(one_block)
spacy.displacy.render(doc_block,style = 'ent', jupyter = True)


# In[106]:


for token in doc_block:
    print(token, token.pos_)


# In[107]:


nouns_verbs = [token.text for token in doc_block if token.pos_ in ('NOUN', 'VERB')]
print(nouns_verbs)


# In[108]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(nouns_verbs)
sum_words = X.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse =True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word','count']
wf_df


# In[109]:


wf_df[0:10].plot.bar(x='word', figsize=(15,10), title = 'Top verbs and nouns')


# In[110]:


from nltk import tokenize
sentence = tokenize.sent_tokenize(' '.join(reviews))
sentence


# In[111]:


sent_df = pd.DataFrame(sentence, columns = ['sentence'])
sent_df


# In[112]:


afinn = pd.read_csv('Afinn.csv', sep =  ',', encoding = 'latin-1')
afinn.shape


# In[113]:


afinn.head()


# In[114]:


from matplotlib.pyplot import imread
get_ipython().run_line_magic('matplotlib', 'inline')


# In[115]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[116]:


sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score


# In[117]:


calculate_sentiment(text = 'amazing')


# In[118]:


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# In[119]:


sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df['word_count'].head(10)


# In[120]:


sent_df.sort_values(by='sentiment_value').tail(10)


# In[121]:


sent_df['sentiment_value'].describe()


# In[122]:


sent_df[sent_df['sentiment_value']>=15].head()


# In[123]:


sent_df['index'] = range(0,len(sent_df))


# In[124]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(sent_df['sentiment_value'])


# In[125]:


plt.figure(figsize = (15,10))
sns.lineplot(y='sentiment_value', x='index', data =sent_df)


# In[126]:


sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')


# In[ ]:




