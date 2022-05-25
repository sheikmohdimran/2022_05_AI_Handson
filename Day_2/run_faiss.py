
import faiss
import numpy as np
import time

embeddings=np.load('output/embeddings.npy')

embeddings.shape
query_vector = embeddings[40:41]

#query_vector=np.expand_dims(query_vector, axis=0)
query_vector.shape

conn_vertex = 16 #32 #Connections for each vertex
dim = 2048  # dimensionality of output features
ef_search = 32
ef_construction = 64

index = faiss.IndexHNSWFlat(dim,conn_vertex)

#Depth of search during build
index.hnsw.efSearch =  ef_search

#Depth of search during search
index.hnsw.efConstruction =  ef_construction

index.add(embeddings)

#Number of nearest neighbors to return
n = 20
start=time.time()
D,I = index.search(query_vector,n)
print(time.time()-start)

#np.in1d(baseline,I)
