from __future__ import print_function
import igraph
import matplotlib.pyplot as plt
import sys
import os
import pylab
import copy
import numpy

"""
Given a path to a graph in .lgl or .gml format (or any other supported by igraph.Graph.Read()),
read in the graph and perform standard analytics on the graph. Keep this class simple and static;
if much reporting is needed, have this spit out intermediate info to files, then let Reporters do
the view work.

Usage: python AnalyzeGraph.py enronGraph.lgl
"""

def _outputNodes():
	return
	
"""
Given a threshold, generates complete, sorted node lists for every common centrality measure.

This outputs the entire list to files, for each measure. Additionally, a global report is compiled, containing
an abbreviated list of the top n nodes, per centrality measure.
"""
def reportGraphStats(g,reportFolder,shown=False):
	gsf = open(reportFolder+"/graphStats.txt","w+")
	rawStats = getGlobalStats(g)
	gsf.write(rawStats+"\r\n")

	#write a diagram of the graph itself (no clustering)
	layout = g.layout_fruchterman_reingold()
	igraph.plot(g,reportFolder+"/"+g["name"].replace(" ","_")+"Graph.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)
	
	#run community_fastgreedy clustering and write it out
	if not g.is_directed():
		print("Computing community_fastgreedy community clusters...")
		of = open(reportFolder+"/fastGreedy.clustering","w+")
		cs = g.community_fastgreedy().as_clustering()
		#write the clusters to file, and save diagrams
		of.write(str(cs))
		of.close()
		if shown:
			igraph.plot(cs,reportFolder+'/fastGreedyClustering.png', layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)
	
	#run all the rest of the clustering algorithms
	print("Computing communities using infomap method...")
	of = open(reportFolder+"/infoMap.clustering","w+")
	cs = g.community_infomap()
	of.write(str(cs))
	of.close()
	if shown:
		igraph.plot(cs,reportFolder+"/infomapClustering.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)

	if not g.is_directed():
		print("Computing communities using eigenvector method...")
		of = open(reportFolder+"/leadingEigenVector.clustering","w+")
		cs = g.community_leading_eigenvector()
		of.write(str(cs))
		of.close()
		if shown:
			igraph.plot(cs,reportFolder+"/eigenvectorClustering.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)

	print("Computing communities using label propagation method...")
	of = open(reportFolder+"/labelPropagation.clustering","w+")
	cs = g.community_label_propagation()
	of.write(str(cs))
	of.close()
	if shown:
		igraph.plot(cs,reportFolder+"/labelPropagationClustering.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)

	if not g.is_directed():
		print("Computing communities using multilevel method...")
		of = open(reportFolder+"/multilevel.clustering","w+")
		cs = g.community_multilevel()
		of.write(str(cs))
		of.close()
		if shown:
			igraph.plot(cs,reportFolder+"/multilevelClustering.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)
	#TODO: I can't get this one to work on my system. It must be a huge matrix blowup; calling this makes the proc memory blow up.
	#print("Computing communities using optimal modularity...")
	#cs = g.community_optimal_modularity()
	#gsf.write(str(cs))
	#if shown:
	#	igraph.plot(cs,reportFolder+"/optimalModularityClustering.png")

	print("Computing communities using edge betweenness...")
	of = open(reportFolder+"/edgeBetweenness.clustering","w+")
	cs = g.community_edge_betweenness().as_clustering()
	of.write(str(cs))
	of.close()
	if shown:
		igraph.plot(cs,reportFolder+"/edgeBetweennessClustering.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)

	if g.is_directed():
		print("Computing communities using walktrap method...")
		of = open(reportFolder+"/walktrap.clustering","w+")
		cs = g.community_walktrap().as_clustering()
		of.write(str(cs))
		of.close()
		if shown:
			igraph.plot(cs,reportFolder+"/walktrapClustering.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)

	if g.is_connected(): #spinglass method requires a connected graph
		print("Computing communities using spinglass method...")
		of = open(reportFolder+"/spinglass.clustering","w+")
		cs = g.community_spinglass()
		of.write(str(cs))
		of.close()
		if shown:
			igraph.plot(cs,reportFolder+"/spinglassClustering.png", layout=layout, bbox = (2048,2048),vertex_label=g.vs["name"],vertex_label_size=9)

	#report top page rank nodes
	print("Computing pageranks...")
	pageList = getMaxPagerankNodes(g,len(g.vs))
	writeCentralities(pageList,reportFolder+"/rawPageRanks.txt",gsf,"PageRank centralities")
	
	#report top degree nodes
	print("Computing degree centralities...")
	if g.is_directed():
		degreeLists = getMaxDegreeNodes_Directed(g,len(g.vs))
		#write the indegree list
		writeCentralities(degreeLists[0],reportFolder+"/rawDegrees_Indegree.txt",gsf,"Indegree centralities")
		#write the outdegree list
		writeCentralities(degreeLists[1],reportFolder+"/rawDegrees_Outdegree.txt",gsf, "Outdegree centralities")
	else:
		degreeList = getMaxDegreeNodes_Undirected(g,len(g.vs))
		writeCentralities(degreeList,reportFolder+"/rawDegrees_Undirected.txt",gsf,"Raw degrees (undirected)")
	
	#report max hub-score nodes
	print("Computing hub scores...")
	hubList = getMaxHubScoreNodes(g,len(g.vs))
	writeCentralities(hubList,reportFolder+'/rawHubScores.txt',gsf,"Hub scores")
	
	#report max authority score nodes
	print("Computing max authority scores...")
	authList = getMaxAuthorityScoreNodes(g,len(g.vs))
	writeCentralities(authList,reportFolder+'/rawAuthorityScores.txt',gsf,"Authority centralities")
	
	#report max eigen centrality nodes
	print("Computing eigenvector centrality scores...")
	eigList = getMaxEigenvectorCentralityNodes(g,len(g.vs))
	writeCentralities(eigList, reportFolder+'./rawEigenvectorScores.txt',gsf,"Eigen vector centralities")
	
	#report the betweenness scores
	print("Computing betweenness scores...")
	btwList = getMaxBetweennessNodes(g,len(g.vs))
	writeCentralities(btwList, reportFolder+'./rawBetweennessScores.txt',gsf,"Betweenness centralities")
	
	print("Analysis complete")
	
	
#This is just a single-purpose output function for a recurring  code pattern in reportGraphStats: given a centrality list of
#tuples (nodeId, centrality value), write them to the outputPath file. Also, writes top 20 results to global stats file.
def writeCentralities(centralityList, outputPath, gsf, centralityMethod="method unknown"):
	#write complete pagerank centrality list to th centrality file
	cf = open(outputPath,"w+")
	cf.writelines([(tup[0]["name"]+", "+str(tup[1])+"\r\n") for tup in centralityList])
	cf.close()
	
	#also write top fifteen centralities to global stats file
	gsf.write("\r\n"+centralityMethod+"\r\n")
	i = 0
	while i < 15 and i < len(centralityList):
		tup = centralityList[i]
		gsf.write(tup[0]["name"]+", "+str(tup[1])+"\r\n")
		i += 1

#Returns k top nodes with highest degree in an undirected graph
def getMaxDegreeNodes_Undirected(g,k):
	if g.is_directed():
		return []

	#deep copy the vertices and sort them by degree
	#TODO: the deep copy is super inefficient; if we need speed, could instead pass around a temp list of vertices to functions like this
	#degList = list(g.vs)
	#degList.sort(key= lambda u : u.degree())
	degList = [(u, u.degree()) for u in g.vs]
	degList.sort(key = lambda tup : tup[1], reverse = True)
	
	#chop the list after k+1
	degList = degList[0:min(len(degList),k)]
	
	return degList

#Returns tuple list (node,pagerank) sorted by max pagerank
def getMaxPagerankNodes(g,k):
	ranks = g.pagerank()
	
	rankList = list(zip(g.vs,ranks))
	#sort list by pagerank
	rankList.sort(key = lambda tup : tup[1], reverse = True)
	
	#chop all items after key
	rankList = rankList[0:min(len(rankList),k)]

	return rankList
	
def getMaxAuthorityScoreNodes(g,k):
	centralities = g.authority_score(scale=False)
	authorityList = list(zip(g.vs,centralities))
	authorityList.sort(key = lambda tup : tup[1], reverse = True)
	authorityList = authorityList[0:min(len(authorityList),k)]
	
	return authorityList

def getMaxEigenvectorCentralityNodes(g,k):
	evals = g.eigenvector_centrality(scale=False)
	evalList = list(zip(g.vs,evals))
	evalList.sort(key = lambda tup : tup[1], reverse = True)
	evalList = evalList[0:min(len(evalList),k)]
	
	return evalList

def getMaxBetweennessNodes(g,k):
	scores = g.betweenness(directed=g.is_directed())
	
	betweenList = list(zip(g.vs,scores))
	betweenList.sort(key = lambda tup : tup[1], reverse = True)
	betweenList = betweenList[0:min(len(betweenList),k)]
	
	return betweenList

def getMaxHubScoreNodes(g,k):
	hubs = g.hub_score(scale=False)
	
	hubList = list(zip(g.vs,hubs))
	hubList.sort(key = lambda tup : tup[1], reverse = True)
	hubList = hubList[0:min(len(hubList),k)]
	
	return hubList

#Returns tuple of two lists: list of nodes sorted by indegree, and list sorted by outdegree
def getMaxDegreeNodes_Directed(g,k):
	if not g.is_directed():
		return []

	#outList = copy.deepcopy(g.vs).sort(key = lambda v : v.outdegree())
	#outList = outList[0:min(len(outList),k)]
	#inList = copy.deepcopy(g.vs).sort(key = lambda v : v.indegree())
	#inList = inList[0:min(len(inList),k)]
	
	indegList = [(u, u.indegree()) for u in g.vs]
	indegList.sort(key = lambda tup : tup[1], reverse = True)
	indegList = indegList[0:min(len(indegList),k)]
	
	outdegList = [(u, u.outdegree()) for u in g.vs]
	outdegList.sort(key = lambda tup : tup[1], reverse = True)
	outdegList = outdegList[0:min(len(outdegList),k)]
	
	return (indegList,outdegList)

"""
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


#Returns simple english string reporting global stats like cluster coefficient, num nodes, etc.
def getGlobalStats(g):
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
"""
	
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
def getGlobalStats(g):

	output = "Global graph stats\r\n"
	
	output += g.summary()
	
	for attribute in g.attributes():
		output += (attribute+": "+g[attribute]+"\r\n")

	isDirected = g.is_directed()
	if isDirected:
		output += "Directed: true\r\n"
	else:
		output += "Directed: false\r\n"
		
	if g.is_weighted():
		output += "Weighted: true\r\n"
	else:
		output = "Weighted: false\r\n"
		
	output += ("Num vertices: "+str(len(g.vs))+"\r\n")
	output += ("Num edges: "+str(len(g.es))+"\r\n")
	if isDirected:
		output += ("Num strong components: "+str(len(g.components(mode=igraph.STRONG)))+"\r\n")
		output += ("Num weak components: "+str(len(g.components(mode=igraph.WEAK)))+"\r\n")
	else:
		output += ("Num components: "+str(len(g.components()))+"\r\n")
	
	"""
	TODO
	print("calculating modularity...")
	output += ("Modularity (unweighted): "+str(g.modularity())+"\n")
	if g.is_weighted():
		output += ("Modularity (weighted): "+str(g.modularity([v["weight"] for v in g.vs]))+"\n")
	"""

	"""
	TODO
	top eigenvector centralities: scond-smalest egienvalue, and largest eigenvalue
	**dont call laplacian() on igraph graph; it will blow up
	"""
	
	print("calculating maxdegree...")
	#get the max degree node
	maxDeg = 0
	for v in g.vs:
		if v.degree() > maxDeg:
			maxV = v
			maxDeg = v.degree()
	
	output += ("Max degree: "+str(maxV.degree())+" ("+v["name"]+")\r\n")
	#avg path length
	print("calculating avg path len...")
	output += ("Average path length: "+str(g.average_path_length())+"\r\n")
	#diameter (longest shortest path)
	print("calculating diameter...")
	output += ("Diameter: "+str(g.diameter())+"\r\n")
	print("calculating components...")

	if not g.is_directed():
		output += ("Global cluster coefficient: "+str(g.transitivity_undirected())+"\r\n")
		output += ("Average local cluster coefficient: "+str(g.transitivity_avglocal_undirected())+"\r\n")

	return output

#plots degree distribution of a graph
def plotDegreeDistribution(g,outputFolder,shown=False):
	#get the raw histogram, then normalize the data to be a probability distribution
	dist = g.degree_distribution()
	xs, ys = zip(*[(left, count) for left, _, count in dist.bins()])

	#normalize the y values to make a probability distribution
	total = 0
	for ct in ys:
		total += ct
	normalized = [(float(ys[i]) / float(total)) for i in range(0,len(ys))]
	ys = tuple(normalized)
	#print("normalized ys: ",ys)

	df = open(outputFolder+"/DegreeDistributionHist.txt","w+")
	df.write(str(dist))
	df.close()
	
	print("max degree is: "+str(max(xs)))
	
	pylab.axis([0,xs[-1]+1,0.0,max(ys)+0.05])
	pylab.bar(xs, ys,width=1.0)
	pylab.title("vertex degree probability distribution")
	pylab.xlabel("degree")
	pylab.ylabel("Px")
	if outputFolder[-1] != "/":
		outputFolder += "/"
	pylab.savefig(outputFolder+"DegreeDistribution.png")
	if shown:
		pylab.show()

def plotPathDistribution2(g,outputFolder):
	xs, ys = zip(*[(left, count) for left, _, count in g.path_length_hist(directed=g.is_directed()).bins()])
	print("xs: "+str(xs))
	print("ys: "+str(ys))
	
	
	
	#pylab.plot(xs,y)
	#plt.show()

def plotPathDistribution(g,outputFolder,shown=False):
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

	pylab.text(0,0,"SOME TEXT")
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

	if shown:
		pylab.show()
	
def usage():
	print("AnalyzeGraph performs basic analytics on a graph given by the passed file.")
	print("Usage: python AnalyzeGraph.py [path to local .graphml or other graph file] [path to output folder for stats reports and graphics]")

if len(sys.argv) < 3:
	print("ERROR insufficient parameters: "+str(len(sys.argv))+str(sys.argv))
	usage()
	exit()
elif not os.path.isfile(sys.argv[1]):
	print("ERROR graph file not found: >"+sys.argv[1]+">")
	usage()
	exit()
elif not os.path.isdir(sys.argv[2]):
	print("ERROR output folder not found: >"+sys.argv[2]+"<")
	usage()
	exit()
else:
	inputFile = sys.argv[1]
	outputFolder = sys.argv[2]
	g = igraph.Graph.Read(inputFile)

	#changes some pylab settings for larger plots
	plt.rcParams["figure.figsize"][0] = 20
	plt.rcParams["figure.figsize"][1] = 15
	
	#prepare the graph by copying the vertex name attributes to a "label" attribute, so names show in plot()
	#print("duplicating names to labels, for plotting...")
	#for v in g.vs:
	#	v["label"] = v["name"]
	
	##generates and writes out most stats to the provided output folder
	reportGraphStats(g,outputFolder,True)
	##plot path dist
	#plotPathDistribution(g,outputFolder,True)
	##plot deg dist
	#plotDegreeDistribution(g,outputFolder,True)
	#TODO: all the spectral stuff. The igraph api is ot workable in this area for large graphs
	#plotEigenvalues(g,outputFolder)
















