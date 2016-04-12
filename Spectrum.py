from __future__ import print_function
import networkx
from numpy import linalg as la


print("Building graph...")
g = networkx.read_graphml("./enronUndirected.graphml")
print("graph:"+str(g))
print("Getting laplacian...")
l = networkx.laplacian_matrix(g)
print("Computing eigenvalues...")
e,v = la.eig(l)




