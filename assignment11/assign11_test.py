from gensim.models import Word2Vec
import numpy as np
import sys
#model = Word2Vec.load("model_fake")
mod =sys.argv[1]
query_file=sys.argv[2]
model = Word2Vec.load(mod + '_fake')
model1=Word2Vec.load(mod + '_real')
#query_file='query'

f=open(query_file)
words=[]
line=f.readline()
while(line !=''):
	row=line.split( )
	words.append(row[0])
	line=f.readline()


#print(words)
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


for i in range(0,len(words),1):
	print('Similar words::Model-Fake(Distance:Cosine )::Word=',words[i])

	print(cosine_distance (model,words[i],model.wv.vocab,5))
	print('Similar words::Model-Real(Distance:Cosine )::Word=',words[i])

	print(cosine_distance (model1,words[i],model1.wv.vocab,5))
	#print('Similar Words (Distance:Euclidean::Word=,',words[i])
	#print(model.most_similar(words[i])[:5])