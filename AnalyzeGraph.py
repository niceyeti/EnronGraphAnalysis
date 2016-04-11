from __future__ import print_function
import igraph
import matplotlib
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


#Returns node with highest degree in an undirected graph
def getMaxDegreeNodes_Undirected(g):
	maxDeg = -1
	maxVertices = []
	#first, find the maximum degree in the graph
	for v in g.vs:
		if v.degree() > maxDeg:
			maxDeg = v.degree()

	for v in g.vs:
		if v.degree() == maxDeg:
			maxVertices.append(v)
			
	print("Max degree nodes: ",maxVertices)
	for v in maxVertices:
		print("deg=",v.degree())
	return maxVertices

def getMaxPagerankNodes(g):
	ranks = g.pagerank()
	maxRank = max(ranks)
	v = []
	i = 0
	for rank in ranks:
		if rank == maxRank:
			v.append(g.vs[i])
			v[-1]["pagerank"] = rank
		i += 1

	return (v,maxRank)

def getMaxAuthorityScoreNodes(g):
	cents = g.authority_score(scale=False)
	maxAuth = max(cents)
	v = []
	i = 0
	for cent in cents:
		if cent == maxAuth:
			v.append(g.vs[i])
			v[-1]["authority"] = cent
		i += 1

	return (v,maxAuth)

def getMaxEigenvectorCentralityNodes(g):
	i = 0
	v = []
	evals = g.eigenvector_centrality(scale=False)
	maxEvcent = max(evals)
	#NOTE this assumes a correspondence between evals in the list and vertex id's, which isn't documented in the py api
	for val in evals:
		if val == maxEvcent:
			maxEvcent = val
			v.append(g.vs[i])
			v[-1]["evcent"] = maxEvcent
		i += 1

	return (v,maxEvcent)

def getMaxBetweennessNodes(g):
	i = 0
	v = []
	scores = g.betweenness(directed=g.is_directed())
	maxScore = max(scores)
	#NOTE this assumes a correspondence between vals in the list and vertex id's, which isn't documented in the py api
	for score in scores:
		if score == maxScore:
			maxScore = score
			v.append(g.vs[i])
			v[-1]["betweenness"] = score
		i += 1

	return (v,maxScore)

def getMaxHubScoreNodes(g):
	i = 0
	v = []
	hubs = g.hub_score(scale=False)
	maxScore = max(hubs)
	#NOTE this assumes a correspondence between vals in the list and vertex id's, which isn't documented in the py api
	for hub in hubs:
		if hub == maxScore:
			maxScore = hub
			v.append(g.vs[i])
			v[-1]["hubscore"] = hub
		i += 1

	return (v,maxScore)

#Returns max in/out degree nodes as a tuple of two lists ([max in], [max out]).
#Multiple nodes may have the same max in/out degree, so the two lists are needed.
def getMaxDegreeNodes_Directed(g):
	maxOutDeg = -1
	maxInDeg = -1
	maxOutVertices = []
	maxInVertices = []

	#first, find the maximum degree in the graph
	for v in g.vs:
		if v.outdegree() > maxOutDeg:
			maxOutDeg = v.outdegree()
		if v.indegree() > maxInDeg:
			maxInDeg = v.indegree()

	for v in g.vs:
		if v.outdegree() == maxOutDeg:
			maxOutVertices.append(v)
		if v.indegree() == maxInDeg:
			maxInVertices.append(v)
			
	print("Max indegree nodes: ",maxInVertices)
	for v in maxInVertices:
		print("deg=",v.indegree())

	print("Max outdegree nodes: ",maxOutVertices)
	for v in maxOutVertices:
		print("deg=",v.outdegree())

	return (maxInVertices,maxOutVertices)

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
	#I'm assuming a correspondence between node id, evals, and evecs from laplacian() and also numpy's eig() functions
	evals, evecs = numpy.linalg.eig( g.laplacian() )
	print("evals: ",str(evals))
	print("evecs: ",str(evecs))
	ids = [v.index for v in g.vs]
	#print("ids: "+str(ids))
	eigs = []
	i = 0
	#pair evals and evecs as tuples; this is bad pyon skill.. should use pack
	while i < len(evals):
		eigs += [(evals[i], evecs[i])]
		i += 1
	#sort the eigenvectors by eigenvalue
	eigs = sorted(eigs, key=lambda tup : tup[0])
	print("eigs: ",eigs)
	#get the eigenvectors, ordered by increasing eigenvalue
	evecs = [eig[1] for eig in eigs]
	print("evecs: ",evecs)

	#plot the eigenvector corresponding with the second-smallest eigenvalue
	pylab.bar(ids, evecs[1])
	pylab.title(str(g["name"])+" eigenvector/vertex values of second-smallest eigenvalue")
	pylab.xlabel("vertex ids")
	pylab.ylabel("eigenvector values")
	pylab.savefig(outputFolder+str(g["name"])+"SecondSmallestEig.png")
	pylab.show()
	
	#plot th eigenvector corresponding with the largest eigenvalue
	pylab.bar(ids, evecs[-1])
	pylab.title(str(g["name"])+" eigenvector/vertex values of largest eigenvalue")
	pylab.xlabel("vertex ids")
	pylab.ylabel("eigenvector values")
	pylab.savefig("./"+str(g["name"])+"LargestEig.png")
	pylab.show()


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
def plotDegreeDist(g,outputFolder):
	#get the raw histogram, then normalize the data to be a probability distribution
	xs, ys = zip(*[(left, count) for left, _, count in 
	g.degree_distribution().bins()])

	#normalize the y values to make a probability distribution
	total = 0
	for ct in ys:
		total += ct
	normalized = [(float(ys[i]) / float(total)) for i in range(0,len(ys))]
	ys = tuple(normalized)
	#print("normalized ys: ",ys)

	pylab.bar(xs, ys)
	pylab.title(g["name"]+"vertex degree probability distribution")
	pylab.xlabel("degree")
	pylab.ylabel("Px")
	if outputFolder[-1] != "/":
		oututFolder += "?"
	pylab.savefig(outputFolder+g["name"]+"DegreeDistribution.png")
	pylab.show()

def plotPathDistribution(g,outputFolder):
	#get the raw histogram, then normalize the data to be a probability distribution
	#hist = g.path_length_hist()
	#print(hist)
	xs, ys = zip(*[(left, count) for left, _, count in g.path_length_hist().bins()])

	#normalize the y values to make a probability distribution
	total = 0
	for ct in ys:
		total += ct
	normalized = [(float(ys[i]) / float(total)) for i in range(0,len(ys))]
	ys = tuple(normalized)
	#print("normalized ys: ",ys)

	pylab.bar(xs, ys)
	pylab.title(g["name"]+"path-length probability distribution")
	pylab.xlabel("path length")
	pylab.ylabel("Px")
	if outputFolder[-1] != "/":
		oututFolder += "?"
	pylab.savefig(outputFolder+g["name"]+"PathLengthDistribution.png")
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

	#stats = getStats(g)
	#row = getRowEntry(g)
	plotDegreeDist(g,outputFolder)
	plotEigenvalues(g,outputFolder)















