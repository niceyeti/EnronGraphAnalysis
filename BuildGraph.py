from __future__ import print_function
from igraph import *
import sys
import os


"""
This class is just a simple encapsulation of the data model construction parameters,
holding information like, should external (outside the @enron network) emails be filtered?
what frequency of emails for nodes (u,v) is sufficient for assigning and edge? etc.
The class doesn't encapsulate any behavior, just data.

@FilterExternal: Toggles whether or not to filter external emails. This seems logical, however
many within the Enron network used external addresses like @aol during the early net years, so it
is best to set this to false and then filter emails by low frequency.
@EdgeFrequencyFilter: The number of emails which must be exchanged (symmetrically/undirected, for simplicity)
in order for (u,v) to have an edge. This is a good de-noise parameter to eliminate all but somewhat-regular
email traffic.
@IsDirected: Whether or not to build a directed or undirected graph
@IsWeighted: Whether or not to track email frequencies per edges (email counts between employees)
@AllowReflexive: Whether or not to allow self-loops (sending email to oneself). We are mostly interested
in the relational, inter-node characteristics of the network, and certain algorithms may require disallowing reflexive
relations, so is best that this is false.
"""
class ModelParams(object):
	def __init__(self,filterExternal=False,edgeFrequencyFilter=1,nodeFrequencyFilter=1,isDirected=False,isWeighted=False,allowReflexive=False):
		self.FilterExternal = filterExternal
		self.EdgeFrequencyFilter = edgeFrequencyFilter
		self.NodeFrequencyFilter = nodeFrequencyFilter
		self.IsDirected = isDirected
		self.IsWeighted = isWeighted
		self.AllowReflexive = allowReflexive

	def ToString(self):
		output = ""
		if self.FilterExternal:
			output += "  filterExternal=True"
		else:
			output += "  filterExternal=False"

		if self.IsDirected:
			output += "  isDirected=True"
		else:
			output += "  isDirected=False"

		if self.IsWeighted:
			output += "  isWeighted=True"
		else:
			output += "  isWeighted=False"

		if self.AllowReflexive:
			output += "  allowReflexive=True"
		else:
			output += "  allowReflexive=False"

		output += ("  EdgeFrequencyFilter="+str(self.EdgeFrequencyFilter))
		output += ("  NodeFrequencyFilter="+str(self.NodeFrequencyFilter))
		
		return output

"""
Given an employee's root folder, returns the email file list within [employee]/sent as absolute paths.
Employee folder must be an absolute path.
"""
def listEmailFiles(employeeFolder):
	allSent = []
	sentFolder = os.path.abspath(employeeFolder+os.sep+"sent")
	if os.path.isdir(sentFolder):
		#get abs path to all sent emails for this employee
		allSent = [os.path.join(sentFolder, sent) for sent in os.listdir(sentFolder) ]
	else:
		print("ERROR sent/ folder not found for "+employeeFolder)
		
	return allSent
		
"""
A resolution function, this takes a list of some employee emails, reads a few of them, and makes a probabilistic assumption
about the sender's email address, returning this sender address as a string. The usage for this function
is that the Enron data is noisy, so its best to evaluate a few emails for a particular sender's address, which will
be used as that node's nuique id.

Sender's email is returned as lower-cased, for case-insensitivity.
"""
def getSenderEmailId(emails):
	addrs = {}
	addr = ""
	curAddr = ""
	i = 0
	while i < 10 and i < len(emails):
		mail = open(emails[i],"r")
		lines = mail.readlines()
		mail.close()
		for line in lines:
			if line.find("From: ") >= 0: # and line.lower().find("@enron.com") >= 0: #the "@enron" was too constraining; removed this since many employees used external addresses
				curAddr = line.strip().split(" ")[1].lower()
				if curAddr in addrs.keys():
					addrs[curAddr] += 1
				else:
					addrs[curAddr] = 1
				break
		i += 1
		
	#return the addr with significant multiplicity
	for key in addrs.keys():
		if addrs[key] >= 3 or i < 3:
			addr = key
			break
	
	return addr.lower()

"""
Given some list of email addresses, removes any that do not contain "@enron".
"""
def filterExternalAddrs(emails):
	#remove external emails
	return [addr for addr in emails if "@enron" in addr]
	
"""
Given an emailFile path, open the email and parse out all the target addresses from the "To: " parameter as
		To: larry.berger@enron.com, raetta.zadow@enron.com, john.buchanan@enron.com, 
			lynn.blair@enron.com, mike.bryant@enron.com, terry.kowalke@enron.com, 
			jean.blair@enron.com, james.carr@enron.com, jodie.floyd@enron.com, 
	Return list of email addresses

	@emailFile: An individual email file in the dataset, corresponding with (sent by) sourceAddr	
	@sourceAddr: The sender address (and assumed node id) of this email
	@params: A ModelParams object for various kinds of filtering

	returns: list of target addresses, lower-cased and filtered according to params. Note the list may still contain dupes.
"""
def listTargetAddresses(emailFile,sourceAddr,params):
	mail = open(emailFile,"r")
	lines = [line.strip() for line in mail.readlines()]
	mail.close()
	addrString = ""
	toSection = False

	for line in lines:
		#"Subject: " comes after any addressees, so break there, we know we have the targets
		if line.find("Subject:") >= 0:
			break
		#first line of "To: " section
		if line.find("To:") == 0:
			addrString += line[3:]
			toSection = True
		#keep consuming "To: " lines until we hit "Subject: " prior to this check.
		elif toSection:
			addrString += (line+",")
		#signals the end of the "To:" section
	
	addrs = [addr.strip() for addr in addrString.lower().replace(" ","").split(",") if len(addr.strip()) > 0]

	#removes self-loops
	if not params.AllowReflexive:
		addrs = filterReflexiveAddrs(addrs,sourceAddr)

	#removes external email addresses (this may be undesirable; see function header)
	if params.FilterExternal:
		addrs = filterExternalAddrs(addrs)

	#print("Found addrs: "+str(addrs))
	#raw_input("debug")
	
	return addrs
	
#Given a list of target email addresses and their source sender's adress, remove sender's address from targets.
#IOW, this removes oneself from one's own target address (when someone sent an email to themselves).
def filterReflexiveAddrs(targets,sourceAddr):
	return [addr for addr in targets if addr != sourceAddr]
	
def resolveAliases(addr,aliasSet,uniqueAddr):
	"""
	This is a single-purpose function. "Vince Kaminsky" in the enron data used many email addresses.
	This function just makes sure that given one of such a person's alias email addresses, we return only one address, so they don't end
	up with multiple nodes on the graph.
	TODO: need to analyze the depth of this problem for other users
	
	"vince.kaminski@enron.com"
	"""
	if addr in aliasSet:
		addr = uniqueAddr
	return addr
	
def filterAliases(addrList,aliasSet,uniqueAddr):
	"""
	Much like the previous function, but for lists: given a list of addresses, get rid of duplicate kaminski aliases, replacing
	them all with only one occurrence of his fixed address. This function is just a fix for the problem of kaminski using multiple addresses.
	
	"vince.kaminski@enron.com"
	"""
	addrs = []
	hasAlias = False
	for addr in addrList:
		if addr in aliasSet:
			hasAlias = True
		else:
			addrs.append(addr)

	#loop removed all kaminski aliases, if there were any; now put his fixed address back in
	if hasAlias:
		addrs.append(uniqueAddr)
	
	return addrs
	
	
"""
Takes in location of EnronData and returns a whom-talks-to-whom frequency graph.
input: enron email dataset location (its assumed folder is structured as: /maildir/[person]/sent)
output: a frequency graph where nodes are enron emailers, and edges are frequencies for how many emails u sends to v

Algorithm:
  -only consider outboxes; this way we eliminate external emails
  -for each outgoing email, increment that edge(s) count(s)
  
  This implementation builds the full enron network in python data structs before flushing them to an igraph network
  
  *Self-links are allowed (self emailing)

	@params: A ModelParams object for toggling how to filter and de-noise the email samples.


"""
def BuildEnronGraph(mailDir,params):
	emailDict = {}  #a nested dictionary of senders (key1) -> target (key2) -> email count
	kaminskiAliases = set(["vkamins@aol.com","vkaminskli@aol.com","vkaminski@aol.com","vkaminskji@aol.com","vkamins@enron.com","kaminski@aol.com","vkaminski@ol.com"])
	layAliases = set(["klay@enron.com","kenneth_lay@enron.com","kenneth.lay@enron.com"])
	missingEmployees = []
	employeeFolders = [os.path.abspath(os.path.join(mailDir,emp)) for emp in os.listdir(mailDir)]
	#print(employeeFolders)
	numEmployees = float(len(employeeFolders))
	i = 0.0
	print("Building graph using parameters: "+params.ToString())

	#foreach employee, parse all of their sent emails
	for emp in employeeFolders:
		i += 1.0
		print("\rprocessing "+emp+"    progress: "+str(int((i/numEmployees) * 100))+"%        ")
		emails = listEmailFiles(emp)
		if len(emails) > 0:
			#probe a few sent emails for the unique sender address
			senderAddr = getSenderEmailId(emails)
			#resolves vkaminsky's multiple addresses
			senderAddr = resolveAliases(senderAddr,kaminskiAliases,"vince.kaminski@enron.com")
			#resolve kenneth lay's multiple addresses
			senderAddr = resolveAliases(senderAddr,layAliases,"kenneth.lay@enron.com")
			#others with aliases/multiple email addrs???
			
			if len(senderAddr) > 0:
				#set up the links from this sender to others they've sent to
				for email in emails:
					#list the target addresses of this email (there may be more than one)
					targets = listTargetAddresses(email,senderAddr,params)
					#Vkaminski used many addresses; this consolidates his addresses into one
					targets = filterAliases(targets,kaminskiAliases,"vince.kaminski@enron.com")
					target = filterAliases(targets, layAliases,"kenneth.lay@enron.com")
					targets = set(targets) #uniquifies the targets
					#add these email peers to this sender's outlinks and track their email counts
					if senderAddr not in emailDict.keys():
						innerDict = {}
						for target in targets:
							innerDict[target] = 1
						emailDict[senderAddr] = innerDict
					else: #append to existing email peer set
						for target in targets:
							#add a frequency entry for this target if not already present
							if target not in emailDict[senderAddr].keys():
								emailDict[senderAddr][target] = 1
							else: #else, target already present, so update target's email count/frequency in inner dictionary
								emailDict[senderAddr][target] += 1
			else:
				print("ERROR sender address empty for emp="+emp)
		else:
			#for reporting: record employees for whom no emails are found
			missingEmployees.append(emp)

	#print("Emaildict: "+str(emailDict))
	#for k in emailDict.keys():
		#print(k+": "+str(emailDict[k]))
		#raw_input("dbg")

	#print(str(emailDict))

	#finally, build the igraph object from the emailDict
	"""
	print("building graph")
	for sender in emailDict.keys():
		for target in emailDict[sender]:
			_addUnweightedLink(sender,target,g)
	"""

	g = convertEmailDictToIGraph(emailDict,params)

	print("graph construction complete")
	#print("g.isDirected="+str(g.is_directed()))
	#print(str([v for v in g.vs]))
	#print(str(g))

	return g

"""
Converts an emailDict to an igraph structure according to the flags held in the @params parameters.

@emailDict: A dictionary of dictionaries. The outer dict contains senderAddr keys mapping to dictionaries. The inner dict
(for a particular senderAddr) contains targetAddrs mapping to email frequency counts.
@params: The graph construction parameters for filtering, etc.

As much as possible, manipulate the emailDict dictionary data structure in python for desired metrics or dataset representations,
then simply output it to an igraph structure as a final step, using this function. Avoid modifying/working on an igrpah graph,
since the igraph api hasn't been very stable nor very efficient. Its much better to do everything in python-land before shoving
everything into igraph just to use their analytics api.

This function also serves another purpose: commentary prefers building the entire igraph Graph all at once as opposed to gradually
iterating over some iterable and calling add_edge(). This is because the overheard of adding an edge to an existing graph is much
higher than simply making all edges/nodes at once. So always use add_edges() and add_vertices() instead of adding
items one at a time.
"""
def convertEmailDictToIGraph(emailDict,params):
	#holds tuples of (sender,target), each with an associated frequency.
	edgeDict = {}
	#now construct the edges, based on logic of params (directed/undirected, weighted/unweighted, filter freq, etc)
	#build an undirected edge list
	if not params.IsDirected:
		#build weighted, undirected edge dictionary: key=(emailAddr1,emailAddr2) -> val=frequency
		for sender in emailDict.keys():
			for target in emailDict[sender].keys():
				#since undirected, check for either key: (sender, target) or (target,sender), symmetrically
				key = (sender,target)
				revKey = (target,sender)
				emailCount = emailDict[sender][target]
				if key in edgeDict.keys():
					edgeDict[key] += emailCount
				elif revKey in edgeDict.keys():
					edgeDict[revKey] += emailCount
				else: #neither key found, so this is a new edge entry
					edgeDict[key] = emailCount
		#post loop: edgeDict contains all undirected edges and their email frequencies
	else:
		#same as prior case, but simpler: just don't check for key symmetry
		for sender in emailDict.keys():
			for target in emailDict[sender].keys():
				#since directed, check and store unique keys, where (senderAddr,targetAddr) != (targetAddr,senderAddr)
				key = (sender,target)
				emailCount = emailDict[sender][target]
				if key in edgeDict.keys():
					edgeDict[key] += emailCount
				else: #key not found, so this is a new edge entry
					edgeDict[key] = emailCount
		#post loop: edgeDict contains all directed edges and their email frequencies
	#post: edgeDict contains keys (senderAddr,destAddr) mapping to email frequencies
	#The construction above unions symmetric key frequencies for undirected graphs, so no further isDirected checks are needed
	#print("edgeDict:"+str(edgeDict))

	#get the edges (the keys) from edgeDict, after filtering low frequency edges, if needed
	edges = [(key[0],key[1],edgeDict[key]) for key in edgeDict.keys() if edgeDict[key] >= params.EdgeFrequencyFilter]
	#post: edges is a list of tuples in the form (sourceAddr,destAddr,frequency)
	#print("edges: "+str(edges))

	#Next construct a frequency table for each node, where emailAddr -> #emails
	#filter low-frequency nodes (nodes to which very few emails have been sent/received)
	nodeDict = {}
	for edge in edges:
		#update sender count
		if edge[0] in nodeDict.keys():
			nodeDict[edge[0]] += edge[2]
		else:
			nodeDict[edge[0]] = edge[2]
		#update receiver count as well (yes, this is correct even for directed graphs)
		if edge[1] in nodeDict.keys():
			nodeDict[edge[1]] += edge[2]
		else:
			nodeDict[edge[1]] = edge[2]

	#intermediate data structures built; now build igraph.Graph itself
	g = Graph(directed=params.IsDirected)
	g["name"] = "Enron email network"
	print("adding vertices...")
	#the union of all senders and targets forms the complete graph node set; note this is done after any edge filtering, so isolated nodes aren't included
	allAddrs = []
	for addr in nodeDict.keys():
		if nodeDict[addr] >= params.NodeFrequencyFilter:
				allAddrs.append(addr)
	#print("all nodes: "+str(nodes))
	#print("adding vertices...")
	g.add_vertices(allAddrs)
	
	print("adding edges...")
	#now we need to filter edges, since we may have dropped nodes which remain in edges[]
	allAddrs = set(allAddrs) #makes a set from the addrList; this will be faster than list for the following lookups)
	#drops edges whose associated nodes were dropped in the previous step
	edges =  [edge for edge in edges if (edge[0] in allAddrs and edge[1] in allAddrs)]
	#add edges to igraph.Graph (not the frequencies, yet)
	edgeList = [(edge[0],edge[1]) for edge in edges]
	#print("edge list: "+str(edgeList))
	g.add_edges(edgeList)

	#update the edge weights; unfortunately the python-igraph api doesn't allow doing this in edge construction, it has to be done iteratively
	print("adding edge weights...")
	if params.IsWeighted:
		for edge in edges:
			edgeId = g.get_eid(edge[0], edge[1], g.is_directed())
			g.es[edgeId]["weight"] = edge[2]
		#print("edges: ",str([edge for edge in g.es]))

	return g

def usage():
	print("Usage: 'python BuildGraph [path to enron dataset /maildir directory] [file location for output graph] [options, listed below]")
	print("The unzipped Enron email dataset contains a 'maildir' directory; provide its path (include '/maildir' in path).")
	print("Options:")
	print("\t--directed: build a directed or undirected version of the enron emails. Undirected graph is default.")
	print("\t--weighted: whether or not to include edge weights (email counts between peers; asymmetric for directed graph). Unweighted graph is default.")
	print("\t--filterExternal: pass this to omit external emails, from outside the @enron email network.")
	print("\t--disallowReflexive: Whether or not to allow reflexive loops (nodes emailing themselves). An edge case, but some algorithms may need this. Such loops are allowed by default.")
	print("\t--edgeFilter=[some int k]: De-noising for edges. Node pairs sharing fewer than k emails will not have an edge. Default is 1 (all edges included).")
	print("\t--nodeFilter=[some int k]: De-noising for nodes. Nodes with fewer than k edges will not be included. This is applied *after* the edge filter. Default is 1 (all nodes included).")
	print("\n\tExample:  python ./BuildGraph ./enron/maildir enronGraph.graphml --filterExternal --edgeFilter=9 --nodeFilter=3 --weighted --directed")
	print("\nComments: Don't use --filterExternal; while seemingly a good idea, many users use/abused external email addresses like '@aol'.")
	
if "help" in sys.argv:
	usage()
elif len(sys.argv) < 3:
	print("ERROR: insufficient parameters: "+str(sys.argv))
	usage()
else:
	#get graph cnstruction params from command line
	isWeighted = "--weighted" in sys.argv
	isDirected = "--directed" in sys.argv
	filterExternal = "--filterExternal" in sys.argv
	allowReflexive = "--disallowReflexive" not in sys.argv

	#set edgeFilter to 1 as default (no filtering)
	edgeFilter= 1
	if len([arg for arg in sys.argv if "--edgeFilter=" in arg]) > 0:
		edgeFilterParam = [arg for arg in sys.argv if "--edgeFilter=" in arg][0]
		edgeFilter = max(1,int(edgeFilterParam.split("=")[1]))
		
	#set nodeFilter to 1 as default (no filtering)
	nodeFilter = 1
	if len([arg for arg in sys.argv if "--nodeFilter=" in arg]) > 0:
		nodeFilterParam = [arg for arg in sys.argv if "--nodeFilter=" in arg][0]
		nodeFilter = max(1,int(nodeFilterParam.split("=")[1]))

	pathToMailDir = sys.argv[1]
	if not os.path.isdir(pathToMailDir):
		print("ERROR: /maildir of Enron dataset not found at location specified: "+pathToMailDir)
		usage()
		exit()

	outputFile = sys.argv[2]
	params = ModelParams(filterExternal, edgeFilter, nodeFilter, isDirected, isWeighted, allowReflexive)
	g = BuildEnronGraph(pathToMailDir,params)
	print("writing graphml-format of graph to "+outputFile)
	g.write_graphml(outputFile)

	#smoke test the generated graph: if it can at least be read back in, I can assume caller can at least read it
	print("Verifying written graph can be read...")
	try:
		g_read = Graph.Read(outputFile)
		if len(g_read.vs) > 0 and len(g_read.es) > 0:
			print("Graph.Read() of output graph succeeded")
		else:
			print("WARN generated graph had no edges or no nodes upon reading back in. Graph build likely unsuccessful.")
	except:
		print("ERROR exception caught reading in graph that was written to "+outputFile+". Graph construction failed.")


"""
Obsolete


#igraph's ugly api for check if node exists, by id. Is there a better way?
def _hasVertex(name,g):
	for node in g.vs:
		#print("name? "+node["name"]+" == "+name)
		if node["name"] == name:
			#print("true")
			return True
	#print("false")
	return False

#Utility for adding unweighted graph links to g for a sender and receiver
def _addUnweightedLink(src,dest,g):
	if src == "" or dest == "":
		print("WARN: empty string passed for node src or dest in addUnweightedLink (edge ignored): src="+src+" dest="+dest)
		return

	hasSrc = _hasVertex(src,g)
	hasDest = _hasVertex(dest,g)
	#add the vertices, as needed
	if not hasSrc:
		g.add_vertex(src)
	if not hasDest:
		g.add_vertex(dest)
	
	if not hasSrc or not hasDest: #if either node didn't already exist, just add new edge
		g.add_edge(src,dest)
	else:  #else need to look up existing edge or create one
		try:
			#igraph api claims get_eid returns -1 if edge is not found, but it actually throws an exception; I just handle both
			edgeId = g.get_eid(src, dest, g.is_directed())
			if edgeId == -1:
				g.add_edge(src,dest)
		except:
			g.add_edge(src,dest)
		# else: nothing, graph is undirected, unweighted for now
"""
