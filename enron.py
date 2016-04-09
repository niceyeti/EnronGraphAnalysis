from __future__ import print_function
from igraph import *
import sys
import os
import matplotlib
import pylab


"""
This class is just a simple encapsulation of the data model construction parameters,
holding information like, should external (outside the @enron network) emails be filtered?
what frequency of emails for nodes (u,v) is sufficient for assigning and edge? etc.
The class doesn't encapsulate any behavior, just data.

@FilterExternal: Toggles whether or not to filter external emails. This seems logical, however
many within the Enron network used external addresses like @aol during the early net years, so it
is best to set this to false and then filter emails by low frequency.
@FrequencyFilter: The number of emails which must be exchanged (symmetrically/undirected, for simplicity)
in order for (u,v) to have an edge. This is a good de-noise parameter to eliminate all but somewhat-regular
email traffic.
@IsDirected: Whether or not to build a directed or undirected graph
@IsWeighted: Whether or not to track email frequencies per edges (email counts between employees)
@AllowReflexive: Whether or not to allow self-loops (sending email to oneself). We are mostly interested
in the relational, inter-node characteristics of the network, and certain algorithms may require disallowing reflexive
relations, so is best that this is false.
"""
class ModelParams(object):
	def __init__(self,filterExternal=False,frequencyFilter=1,isDirected=False,isWeighted=False,allowReflexive=False):
		self.FilterExternal = filterExternal
		self.FrequencyFilter = frequencyFilter
		self.IsDirected = isDirected
		self.IsWeighted = isWeighted
		self.AllowReflexive = allowReflexive

#Print wrapper so both 2.7 and 3.0 print functions can be used.
def _print(*args):
	print(*args)
	#argList = [arg for arg in args]
	#if sys.version_info[0] < 3:
	#	print args
	#else:
	#	print(args)

"""
Given an employee's root folder, returns the email file list within [employee]/sent as absolute paths.
Employee folder must be an absolute path.
"""
def listEmailFiles(employeeFolder):
	allSent = []
	sentFolder = os.path.abspath(employeeFolder+os.sep+"sent")
	#_print(sentFolder)
	if os.path.isdir(sentFolder):
		#_print("Parsing mail for "+sentFolder)
		#get abs path to all sent emails for this employee
		allSent = [os.path.join(sentFolder, sent) for sent in os.listdir(sentFolder) ]
	else:
		_print("ERROR sent/ folder not found for "+employeeFolder)
		
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
def _filterExternalAddrs(emails):
	#remove external emails
	addrs = [addr for addr in emails if "@enron" in addr]
	#removed = [addr for addr in emails if "@enron" not in addr]
	#if removed != None and len(removed) > 0:
	#	_print("filtered external emails: ",removed)

	return addrs
	
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

	#remove self-loops
	if not params.AllowReflexive:
		addrs = _filterReflexiveAddrs(addrs,sourceAddr)

	#remove external emails
	if params.FilterExternal:
		addrs = _filterExternalAddrs(addrs)

	#print("Found addrs: "+str(addrs))
	#raw_input("debug")
	
	return addrs
	
#Given a list of target email addresses and their source sender's adress, remove sender's address from targets.
#IOW, this removes oneself from one's own target address (when someone sent an email to themselves).
def _filterReflexiveAddrs(targets,sourceAddr):
	return [addr for addr in targets if addr != sourceAddr]

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
def BuildStaticGraph(mailDir,params):
	g = Graph()
	g["MyName"] = "Enron static email network"
	emailDict = {}  #a nested dictionary of senders (key1) -> target (key2) -> email count
	missingEmployees = []
	employeeFolders = [os.path.abspath(os.path.join(mailDir,emp)) for emp in os.listdir(mailDir)]
	#_print(employeeFolders)
	numEmployees = float(len(employeeFolders))
	i = 0.0
	#foreach employee, parse all of their sent emails
	for emp in employeeFolders:
		#if i > 10.0:
		#	break
		i+=1.0
		_print("\rprocessing "+emp+"  progress: "+str(int((i/numEmployees) * 100))+"%        ")
		emails = listEmailFiles(emp)
		if len(emails) > 0:
			#probe a few sent emails for the unique sender address
			senderAddr = getSenderEmailId(emails)
			if len(senderAddr) > 0:
				#set up the links
				for email in emails:
					#list the target addresses of this email (there could be more than one)
					targets = listTargetAddresses(email,senderAddr,params)
					targets = set(targets) #uniquify the targets
					#add these email peer to this sender's outlink list as a set.
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
				#_print("sender: "+senderAddr)
			else:
				print("ERROR sender address empty for emp="+emp)
		else:
			#for reporting: record employees for whom no emails are found
			missingEmployees.append(emp)

	#_print("Emaildict: "+str(emailDict))
	#for k in emailDict.keys():
		#_print(k+": "+str(emailDict[k]))
		#raw_input("dbg")

	print(str(emailDict))

	#finally, build the igraph object from the emailDict
	"""
	print("building graph")
	for sender in emailDict.keys():
		for target in emailDict[sender]:
			_addUnweightedLink(sender,target,g)
	"""

	#g = _convertDictToIGraph(emailDict,params)

	#Now convert the python data structures to an igraph. An important system/api note is that online
	#commentary prefers building the entire igraph Graph all at once as opposed to iterating over some iterable
	#and calling add_edge(). This is because the overheard of adding an edge to an existing graph is much
	#higher than simply making all edges/nodes at once. So always use add_edges() and add_vertices() instead of adding
	#items one at a time.

	#the union of all senders and targets forms the complete graph node set
	allAddrs = [sender for sender in emailDict.keys()] + [target for key in emailDict.keys() for target in emailDict[key]]
	print("all addrs: "+str(allAddrs))
	nodes = [set(allAddrs)] #uniquify the set of addresses/node-ids
	print("all nodes: "+str(nodes))
	print("adding vertices...")
	g.add_vertices(nodes)
	
	



	#add_vertices(n) where n is a number of list of strings for new vertex names
	#add_edges() where passed are tuples of nodes ids or names of endpoints

	print("graph construction completed")
	return g

#igraph's disgustingly inefficient api for check if node exists, by id. Is there a better way?
def _hasVertex(name,g):
	for node in g.vs:
		#_print("name? "+node["name"]+" == "+name)
		if node["name"] == name:
			#_print("true")
			return True
	#_print("false")
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

#enronRootDir = "./maildir"
#_print("sep="+os.sep)
"""
enronRootDir = "./testdir"
g = BuildStaticGraph(enronRootDir,False)
g.write_gml("testGraph.gml")
g = Graph.Read("testGraph.gml")
_print(g)
"""

params = ModelParams(filterExternal=False,frequencyFilter=1,isDirected=False,isWeighted=False,allowReflexive=False)

enronRootDir = "./testdir"
g = BuildStaticGraph(enronRootDir,params)
print("writing graph to file...")
g.write_lgl("enronGraph.lgl")
#_print("vertices: "+str([v for v in g.vs]))
#_print("edges: "+str([e for e in g.es]))
#g.write_gml("enronGraph.gml")
_print("Verifying written graph can be read...")
g = Graph.Read("enronGraph.lgl")
_print(g)

#g = Graph()
#g.add_vertices(3)
#g.add_edges([(0,1),(1,2),(2,0)])
#g.write_gml("dummy.gml")
#_print(str(g))
