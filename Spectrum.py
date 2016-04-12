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
w,v = sp.linalg.eigs(l.A)
print("done")




