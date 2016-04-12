from __future__ import print_function
import networkx
from numpy import linalg as la
from scipy import sparse as sp

print("Building graph...")
g = networkx.read_graphml("./enronUndirected.graphml")
print("graph of "+str(g.number_of_nodes())+" nodes:"+str(g))
print("Getting laplacian...")
l = networkx.normalized_laplacian_matrix(g)
#print(str(l))
print("Computing eigenvalues...")
#e,v = la.eig(l.A) #explodes
w,v = sp.linalg.eigs(l)
print("done")
#blah

"""

Create an G{n,m} random graph and compute the eigenvalues.
Requires numpy and matplotlib.

import networkx as nx
import numpy.linalg
import matplotlib.pyplot as plt

n = 1000 # 1000 nodes
m = 5000 # 5000 edges
G = nx.gnm_random_graph(n,m)

L = nx.normalized_laplacian_matrix(G)
e = numpy.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))
print("Smallest eigenvalue:", min(e))
plt.hist(e,bins=100) # histogram with 100 bins
plt.xlim(0,2)  # eigenvalues between 0 and 2
plt.show()
"""

