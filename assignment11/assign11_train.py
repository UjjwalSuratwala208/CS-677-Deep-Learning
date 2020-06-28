import numpy as np
import pandas as pd
from numpy import linalg as LA
import sys
from gensim.models import Word2Vec

# In[7]:
NUM_WORDS=20000

def cosine_distance (model, word,target_list , num,word_vectors) :
	cosine_dict ={}
	word_list = []
	a =word_vectors[word]
	#for item in target_list :
	for item, i in target_list.items():
		if item != word :
			if i>=NUM_WORDS:
				continue
			
			b =word_vectors [item]
			cos_sim = np.dot(a, b)/(LA.norm(a)*LA.norm(b))
			cosine_dict[item] = cos_sim
	dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
	for item in dist_sort:
		word_list.append((item[0], item[1]))
	return word_list[0:num]

f_n=sys.argv[1]
m_n=sys.argv[2]
data = pd.read_csv(f_n)

#input=sys.argv[1]
# In[8]:

data.head()


# In[9]:

data["text"] = data["title"].map(str) + data["text"]



# In[10]:

data.head()


# In[11]:

data = data.loc[:,['text','label']]
print(data.shape)
dat=data['label']=='FAKE'
bat=data['label']=='REAL'
#print(data[dat].shape)
# In[12]:
dats=data[dat]
bata=data[bat]

data.head()


# In[13]:


train_data=dats


# In[25]:



# In[7]:



# In[8]:



# In[15]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# In[16]:

texts = train_data.text
btexts=bata.text

print("texts",len(texts))

sent = [row.split() for row in texts]
bsent = [row.split() for row in btexts]

#print(sent[0])

model = Word2Vec(sent, min_count=1,size= 50,workers=3, window =3, sg = 1)
model1 = Word2Vec(bsent, min_count=1,size= 50,workers=3, window =3, sg = 1)

#print(model.most_similar('Hillary')[:5])


from numpy import linalg as LA
def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = np.dot(a, b)/(LA.norm(a)*LA.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]
# only get the unique Maker_Model

# Show the most similar Mercedes-Benz SLK-Class by cosine distance 
#print(cosine_distance (model,'Hillary',model.wv.vocab,5))

model.save(m_n + '_fake')

model1.save(m_n +'_real')











#model = Word2Vec(sent, min_count=1,size= 50,workers=3, window =3, sg = 1)