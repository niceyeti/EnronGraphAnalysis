from __future__ import print_function
import igraph
import matplotlib.pyplot as plt
import sys
import os
import pylab
import numpy

"""
Given a path to a graph in .lgl or .gml format (or any other supported by igraph.Graph.Read()),
read in the graph and perform standard analytics on the graph. Keep this class simple and static;
if much reporting is needed, have this spit out intermediate info to files, then let Reporters do
the view work.

Usage: python AnalyzeGraph.py enronGraph.lgl
"""

def compDegree(u):
	return u.degree()

#Returns k top nodes with highest degree in an undirected graph
def getMaxDegreeNodes_Undirected(g,k):
	if g.is_directed():
		return []

	#deep copy the vertices and sort them by degree
	#TODO: the deep copy is super inefficient; if we need speed, could instead pass around a temp list of vertices to functions like this
	degList = copy.deepcopy(g.vs)
	degList.sort(key= lambda u : u.degree())
	
	#chop the list after k+1
	degList = degList[0:min(len(degList),k)]
	
	return degList

def getMaxPagerankNodes(g,k):
	ranks = g.pagerank()
	
	rankList = list(zip(g.vs,ranks))
	#sort list by pagerank
	rankList.sort(key = lambda tup : tup[1])
	
	#chop all items after key
	rankList = rankList[0:min(len(rankList),k)]

	return rankList
	
def getMaxAuthorityScoreNodes(g,k):
	centralities = g.authority_score(scale=False)
	authorityList = list(zip(g.vs,centralities))
	authoriyList.sort(key = lambda tup : tup[1])
	authorityList = authorityList[0:min(len(authorityList),k)]
	
	return authorityList

def getMaxEigenvectorCentralityNodes(g):
	evals = g.eigenvector_centrality(scale=False)
	evalList = list(zip(g.vs,evals))
	evalList.sort(key = lambda tup : tup[1])
	
	return evalList

def getMaxBetweennessNodes(g,k):
	scores = g.betweenness(directed=g.is_directed())
	
	betweenList = list(zip(g.vs,scores))
	betweenList.sort(key = lambda tup : tup[1])
	betweenList = betweenList[0:min(len(betweenList,k))]
	
	return betweenList

def getMaxHubScoreNodes(g,k):
	hubs = g.hub_score(scale=False)
	
	hubList = list(zip(g.vs,hubs))
	hubList.sort(key = lambda tup : tup[1])
	hubList = hubList[0:min(len(hubList,k))]
	
	return hubList

#Returns tuple of two lists: list of nodes sorted by indegree, and list sorted by outdegree
def getMaxDegreeNodes_Directed(g,k):
	if not g.is_directed():
		return []

	outList = copy.deepcopy(g.vs).sort(key = lambda v : v.outdegree())
	outList = outList[0:min(len(outList),k)]
	
	inList = copy.deepcopy(g.vs).sort(key = lambda v : v.indegree())
	inList = inList[0:min(len(inList),k)]
	
	return (inList,outList)

#Returns a formatted string of centrality measures.
def getCentralities(g):
	s = ""
	#degree centrality; if directed, get highest in/out degree nodes; else get node with highest undirected degree
	if g.is_directed():
		maxNodes = getMaxDegreeNodes_Directed(g)
		s += ("Max indegree and vertex: "+str(maxNodes[0][0].indegree())+str(maxNodes[0])+"\r\n")
		s += ("Max outdegree and vertex: "+str(maxNodes[1][0].outdegree())+str(maxNodes[1])+"\r\n")
	else:
		maxNodes = getMaxDegreeNodes_Undirected(g)
		s += ("Max degree and vertices:"+str(maxNodes[0].degree())+str(maxNodes)+"\r\n")
	#print("Max nodes:", maxNodes)

	#get the max eigenvalue centrality
	maxNodes = getMaxEigenvectorCentralityNodes(g)
	s += ("Max eigenvector centrality value and nodes: "+str(maxNodes[1])+"  "+str(maxNodes[0])+"\r\n")

	#get the max pagerank centrality node
	maxNodes = getMaxPagerankNodes(g)
	s += ("Max pagerank and node id: "+str(maxNodes[1])+"  "+str(maxNodes[0]) +"\r\n")

	#get the authority scores
	maxNodes = getMaxAuthorityScoreNodes(g)
	s += ("Max authority score and node id: "+str(maxNodes[1])+"  "+str(maxNodes[0]) +"\r\n")

	#get the max hub score nodes
	maxNodes = getMaxHubScoreNodes(g)
	s += ("Max hub score and node id: "+str(maxNodes[1])+"  "+str(maxNodes[0]) +"\r\n")

	#get the max betweenness score nodes
	maxNodes = getMaxBetweennessNodes(g)
	s += ("Max betweenness score and node id: "+str(maxNodes[1])+"  "+str(maxNodes[0]) +"\r\n")

	return s

#Creates in the table row entry for graph in problem 2: num edges, nodes, evals, etc as ampersand-delimited values
def getRowEntry(g):
	n = len(g.vs)
	m = len(g.es)
	maxDegree = max(g.degree())
	minDegree = min(g.degree())
	diameter = g.diameter(directed=g.is_directed())
	avgPathLen = g.average_path_length(directed=g.is_directed())
	g_clusterCoef = g.transitivity_undirected()

	laplacianM = g.laplacian(normalized=False)
	evals = numpy.linalg.eigvals(laplacianM)
	evals.sort()
	print("Sorted evals: ",evals)
	#get second smallest eigen value
	lambda_2 = evals[1]
	#get the largest eigenvalue
	lambda_n = evals[-1]

	s = str(n) + "  &  " + str(m) + "  &  " + str(minDegree) + "  &  " + str(maxDegree) + "  &  " + str(avgPathLen) + "  &  " + str(diameter) + "  &  " + str(g_clusterCoef) + "  &  " + str(lambda_2) + "  &  " + str(lambda_n)

	return s

#Plots the eigenvector's of the largest and second smallest eigenvalues against node id's, in two plots.
def plotEigenvalues(g,outputFolder):
	if g.is_directed():
		print("ERROR directed graph passed to plotEigenvalues; laplacian requires undirected graph")
		return

	#I'm assuming an index-correspondence between node id, evals, and evecs from laplacian() and also numpy's eig() functions
	print("getting laplacian...")
	laplacian = g.laplacian()
	print("computing eigenvalues/vectors...")
	evals, evecs = numpy.linalg.eig( laplacian )
	#print("evals: ",str(evals))
	#print("evecs: ",str(evecs))
	ids = [v.index for v in g.vs]
	#print("ids: "+str(ids))
	#zip the values into a list of 3-tuples as (nodeId, eigenValue, eigenVector)
	eigList = list(zip(ids,evals,evecs))
	#sort the list by eigenvalue in increasing order
	eigList.sort(key = lambda tup : tup[1])
	#print("eiglist: "+str(eigList))
	
	print("plotting...")
	#plot the eigenvector corresponding with the second-smallest eigenvalue
	pylab.bar(ids, eigList[1][2])
	#pylab.axis(ids[0],ids[-1],)
	pylab.title(" eigenvector/vertex values of second-smallest eigenvalue")
	pylab.xlabel("vertex ids")
	pylab.ylabel("eigenvector values")
	if outputFolder[-1] != "/":
		outputFolder += "/"
	pylab.savefig(outputFolder+"SecondSmallestEig.png")
	pylab.show()
	
	#plot the eigenvector corresponding with the largest eigenvalue
	pylab.bar(ids, eigList[-1][2])
	#pylab.xlim(ids[0],ids[-1])
	pylab.title("eigenvector/vertex values of largest eigenvalue")
	pylab.xlabel("vertex ids")
	pylab.ylabel("eigenvector values")
	pylab.savefig(outputFolder+"LargestEig.png")
	pylab.show()
	
	"""
	print("getting eigenvalues/vectors...")
	evals, evecs = numpy.linalg.eig( g.laplacian() )
	ids = [v.index for v in g.vs]
	
	eigs = []
	i = 0
	#pair evals and evecs as tuples; this is bad pyon skill.. should use pack
	while i < len(evals):
		eigs += [(evals[i], evecs[i])]
		i += 1
	#sort the eigenvectors by eigenvalue
	eigs = sorted(eigs, key=lambda tup : tup[0])
	#print("eigs: ",eigs)
	#get the eigenvectors, ordered by increasing eigenvalue
	evecs = [eig[1] for eig in eigs]
	#print("evecs: ",evecs)

	#plot the eigenvector corresponding with the second-smallest eigenvalue
	pylab.bar(ids, evecs[1])
	pylab.title(" eigenvector/vertex values of second-smallest eigenvalue")
	pylab.xlabel("vertex ids")
	pylab.ylabel("eigenvector values")
	if outputFolder[-1] != "/":
		outputFolder += "/"
	pylab.savefig(outputFolder+"SecondSmallestEig.png")
	pylab.show()
	
	#plot the eigenvector corresponding with the largest eigenvalue
	pylab.bar(ids, evecs[-1])
	pylab.title("eigenvector/vertex values of largest eigenvalue")
	pylab.xlabel("vertex ids")
	pylab.ylabel("eigenvector values")
	pylab.savefig(outputFolder+"LargestEig.png")
	pylab.show()
	"""
	
#prints graph stats by column: name,directedness(d/u),numlinks,nvertices,maxdegree, etc
def getStats(g):
	output = "unknown"
	if "name" in g.attributes():
		output = g["name"]

	isDirected = g.is_directed()
	if isDirected:
		output += ",d"
	else:
		output += ",u"

	output += (","+str(len(g.vs)))
	output += (","+str(len(g.es)))
	if isDirected:
		output += (",comps="+str(len(g.components(mode=igraph.STRONG)))+"(strong)")
		output += ("/"+str(len(g.components(mode=igraph.WEAK)))+"(weak)")
	else:
		output += (",comps="+str(len(g.components())))
	print("calculating maxdegree...")
	output += (","+str(g.maxdegree()))
	#avg path length
	print("calculating avg path len...")
	output += (","+str(g.average_path_length()))
	#diameter (longest shortest path)
	print("calculating diameter...")
	output += (","+str(g.diameter()))
	print("calculating components...")

	output += (",g_clstr="+str(g.transitivity_undirected())+"(global)")
	output += (",avg_clsr="+str(g.transitivity_avglocal_undirected())+"(avg local)")

	#shoved this in for latex table formatting
	output = output.replace(","," & ")

	return output

#plots degree distribution of a graph
def plotDegreeDistribution(g,outputFolder):
	#get the raw histogram, then normalize the data to be a probability distribution
	xs, ys = zip(*[(left, count) for left, _, count in g.degree_distribution().bins()])

	#normalize the y values to make a probability distribution
	total = 0
	for ct in ys:
		total += ct
	normalized = [(float(ys[i]) / float(total)) for i in range(0,len(ys))]
	ys = tuple(normalized)
	#print("normalized ys: ",ys)

	print("max degree is: "+str(max(xs)))
	
	pylab.axis([0,xs[-1]+1,0.0,max(ys)+0.05])
	pylab.bar(xs, ys,width=1.0)
	pylab.title("vertex degree probability distribution")
	pylab.xlabel("degree")
	pylab.ylabel("Px")
	if outputFolder[-1] != "/":
		outputFolder += "/"
	pylab.savefig(outputFolder+"DegreeDistribution.png")
	pylab.show()

def plotPathDistribution2(g,outputFolder):
	xs, ys = zip(*[(left, count) for left, _, count in g.path_length_hist(directed=g.is_directed()).bins()])
	print("xs: "+str(xs))
	print("ys: "+str(ys))
	
	
	
	#pylab.plot(xs,y)
	#plt.show()

def plotPathDistribution(g,outputFolder):
	#get the raw histogram, then normalize the data to be a probability distribution
	#hist = g.path_length_hist()
	#print(hist)
	xs, ys = zip(*[(int(left), count) for left, _, count in g.path_length_hist(directed=g.is_directed()).bins()])

	#normalize the y values to make a probability distribution
	total = 0
	for ct in ys:
		total += ct
	normalized = [(float(ys[i]) / float(total)) for i in range(0,len(ys))]
	ys = tuple(normalized)
	#print("normalized ys: ",ys)

	pylab.text(0,0,"BLAH")
	pylab.axis([0,xs[-1]+1,0.0,max(ys)+0.05])
	pylab.bar(xs, ys,width=1.0)
	#pylab.axis([0,xs[-1],0.0,ys[-1]])
	#pylab.xlim(0,max(max(xs),1))
	pylab.title("path-length probability distribution")
	pylab.xlabel("path length")
	pylab.ylabel("Px")
	if outputFolder[-1] != "/":
		outputFolder += "/"
	pylab.savefig(outputFolder+"PathLengthDistribution.png")
	pylab.show()
	
def usage():
	print("AnalyzeGraph performs basic analytics on a graph given by the passed file.\n")
	print("Usage: python AnalyzeGraph.py [path to local .gml, .lgl or other graph file] [path to output dir for reports, graphics]\n")

if len(sys.argv) < 3:
	print("ERROR insufficient parameters: "+str(len(sys.argv))+str(sys.argv))
	usage()
	exit()
elif not os.path.isfile(sys.argv[1]):
	print("ERROR graph file not found: "+sys.argv[1])
	usage()
	exit()
elif not os.path.isdir(sys.argv[2]):
	print("ERROR output folder not found: "+sys.argv[2])
	usage()
	exit()
else:
	inputFile = sys.argv[1]
	outputFolder = sys.argv[2]
	g = igraph.Graph.Read(inputFile)

	#changes some pylab settings for larger plots
	plt.rcParams["figure.figsize"][0] = 12
	plt.rcParams["figure.figsize"][1] = 9
	
	
	#stats = getStats(g)
	#row = getRowEntry(g)
	#plotPathDistribution(g,outputFolder)
	#plotDegreeDistribution(g,outputFolder)
	plotEigenvalues(g,outputFolder)
















