
import numpy as np
import sys

#################
### Read data ###

f = open(sys.argv[1])
data1 = np.loadtxt(f)


onearray = np.ones((data1.shape[0],1))
data1 = np.append(data1,onearray,axis=1)

train = data1[:,1:]
trainlabels = data1[:,0]

print("train=",train)
print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]

#hidden_nodes = int(sys.argv[3])

hidden_nodes = 3
k=1
k=int(sys.argv[3])
##############################
### Initialize all weights ###

w = np.random.rand(hidden_nodes)
#w = np.ones(hidden_nodes)
print("w=",w)

#check this command
#W = np.zeros((hidden_nodes, cols), dtype=float)
#W = np.ones((hidden_nodes, cols), dtype=float)
W = np.random.rand(hidden_nodes, cols)

print("W=",W)

epochs = 200
eta = 0.001
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
print("hidden_layer=",hidden_layer)
print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
#sigmoid = lambda x: np.sign(x)
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
print("hidden_layer=",hidden_layer)
print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
#print("obj=",obj)

#obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))

print("Initial Obj=",obj)
data111=np.array([i for i in range(rows)])
###############################
### Begin gradient descent ####
#exit()
while(i < epochs):
	
	#Update previous objective
	prevobj = obj
	#print(train)
	#Calculate gradient update for final layer (w)
	#dellw is the same dimension as w
	dellw, dells, dellu, dellv=0, 0, 0, 0
	#rows = train.shape[0]
	for mk in range(0, rows):
		np.random.shuffle(data111)
		cur=data111[0]
		dellw = (np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*hidden_layer[cur,:]
		for j in range(1, k):
            		cur=data111[j]
            		dellw += (np.dot(hidden_layer[cur,:],np.transpose(w))-trainlabels[cur])*hidden_layer[cur,:]

	    #Update w
	    #print(dellw)
		w = w - eta*dellw

        #print("dellf=",dellf)
	
	    #Calculate gradient update for hidden layer weights (W)
	    #dellW has to be of same dimension as W

	    #Let's first calculate dells. After that we do dellu and dellv.
	    #Here s, u, and v are the three hidden nodes
	    #dells = df/dz1 * (dz1/ds1, dz1,ds2)
	    #print((hidden_layer[0,0])*(1-hidden_layer[0,0])*train[0])
		cur=data111[0]
		dells = np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[0] * (hidden_layer[cur,0])*(1-hidden_layer[cur,0])*train[0]
		for j in range(1, k):
                	cur=data111[j]
                	dells += np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[0] * (hidden_layer[cur,0])*(1-hidden_layer[cur,0])*train[j]
	    #print(dells)
	


	    #TODO: dellu = ?
		dellu = np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[1] * (hidden_layer[cur,1])*(1-hidden_layer[cur,1])*train[0]
		for j in range(1, k):
			cur=data111[j]
			dellu += np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[1] * (hidden_layer[cur,1])*(1-hidden_layer[cur,1])*train[cur]


	    #TODO: dellv = ?
		dellv = np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[2] * (hidden_layer[cur,2])*(1-hidden_layer[cur,2])*train[cur]
		for j in range(1, k):
			cur=data111[j]
			dellv += np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[2] * (hidden_layer[cur,2])*(1-hidden_layer[cur,2])*train[cur]


	    #TODO: Put dells, dellu, and dellv as rows of dellW
	    #dellW=[dells,dellu,dellv]
		dellW=np.array([dells, dellu, dellv])
	    #print(dellW)
	
	    #Update W
		W = W - eta*dellW
	#print(W)
	#print(w)
	#exit()
	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))
	#print("hidden_layer=",hidden_layer)

	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
	#print("hidden_layer=",hidden_layer)

	output_layer = np.matmul(hidden_layer, np.transpose(w))
	#print("output_layer=",output_layer)

	obj = np.sum(np.square(output_layer - trainlabels))
	print("obj=",obj)
	
	i = i + 1
	#print("Objective=",obj)
	

#predictions = np.sign(np.matmul(test, np.transpose(w)))
hidden_layer = np.matmul(test, np.transpose(W))
predictions = (np.matmul(sigmoid(hidden_layer),np.transpose(w)))
predictions1 = np.sign(predictions)
print(predictions1)
print(w)


