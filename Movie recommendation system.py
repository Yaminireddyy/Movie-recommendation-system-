#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


credits_df=pd.read_csv("credits.csv")
movies_df=pd.read_csv("movies.csv")


# In[3]:


movies=movies_df.merge(credits_df,on='title')


# In[4]:


movies_pr=movies[["genres","id_x","keywords","overview","popularity","revenue","title","cast","crew"]]


# In[5]:


movies_pr.iloc[0].genres


# In[6]:


#Removing null values
movies_pr.isnull().sum()
movies_pr=movies_pr.dropna(axis=0)



# In[7]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[8]:


res=[]
for i in movies_pr['genres']:
    res.append(convert(i))
movies_pr["genres"]=res   


# In[9]:


movies_pr["keywords"]=movies_pr["keywords"].apply(convert)


# In[10]:


def convert_cast(obj):
    l=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!=3:
            l.append(i['name'])
            count+=1
        else:
            break
    return l
    


# In[11]:


movies_pr["cast"]=movies_pr["cast"].apply(convert_cast)




# In[12]:


def director(x):
    lst=[]
    for i in ast.literal_eval(x):
        if i['job']=='Director':
            lst.append(i['name'])
            break
    return lst


# In[13]:


movies_pr['crew']=movies_pr['crew'].apply(director)


# In[14]:


def overview(s):
    l=s.split()
    return l


# In[15]:


movies_pr['overview']=movies_pr['overview'].apply(overview)


# In[16]:


movies_pr["genres"]=movies_pr["genres"].apply(lambda x:[i.replace(" ","")for i in x])
movies_pr["keywords"]=movies_pr["keywords"].apply(lambda x:[i.replace(" ","")for i in x])
movies_pr["crew"]=movies_pr["crew"].apply(lambda x:[i.replace(" ","")for i in x])
movies_pr["cast"]=movies_pr["cast"].apply(lambda x:[i.replace(" ","")for i in x])



# In[17]:


movies_pr["tags"]=movies_pr["genres"]+movies_pr["keywords"]+movies_pr["crew"]+movies_pr["cast"]


# In[18]:


movies_newpr=movies_pr[['id_x','title','tags']]


# In[19]:


movies_newpr['tags']=movies_newpr['tags'].apply(lambda x:" ".join(x))


# In[20]:


movies_newpr['tags']=movies_newpr['tags'].apply(lambda x:x.lower())


# In[21]:


ps=PorterStemmer()


# In[22]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
    


# In[23]:


movies_newpr['tags']=movies_newpr['tags'].apply(stem)


# In[24]:


cv=CountVectorizer(max_features=5000,stop_words='english')


# In[25]:


vectors=cv.fit_transform(movies_newpr['tags']).toarray()


# In[26]:


similarity=cosine_similarity(vectors)


# In[27]:


def recommend(movie):
    movie_index=movies_newpr[movies_newpr['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(movies_newpr.iloc[i[0]].title)


# In[28]:


movieslist=recommend('Batman Begins')


# In[29]:


import pickle


# In[30]:


pickle.dump(movies_newpr.to_dict(),open('movies.pkl','wb'))


# In[31]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:




