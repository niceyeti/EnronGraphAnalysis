#A bunch of code for playing around with constructing graphs and graph metrics from the Enron email data set
from igraph import *
import sys
import os
import matplotlib
import pylab

#Print wrapper so both 2.7 and 3.0 print functions can be used.
def _print(*args):
	argList = [arg for arg in args]
	if sys.version_info[0] < 3:
		print argList
	else:
		print(argList)

"""
Given an employee's root folder, returns the email file list within [employee]/sent as absolute paths.
Employee folder must be an absolute path.
"""
def listEmailFiles(employeeFolder):
	allSent = []
	sentFolder = os.path.abspath(employeeFolder+os.sep+"sent")
	_print(sentFolder)
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
				curAddr = line.strip().split(" ")[1]
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
	
	return addr
	

"""
Given some list of email addresses, removes any that do not contain "@enron".
External addresses are printed, for debugging
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
	
	@filterExternal: True by default, this specifies that emails outside of the enron network should be
	excluded (eg, address does not contain "@enron")
"""
def listTargetAddresses(emailFile,filterExternal=True):
	mail = open(emailFile,"r")
	lines = [line.strip() for line in mail.readlines()]
	mail.close()
	addrString = ""
	toSection = False
	for line in lines:
		if line.find("Subject:") >= 0:
			break
		if line.find("To:") == 0:
			addrString += line[3:].strip()
			toSection = True
		elif toSection:
			addrString += (line.strip()+",")
		#signals the end of the "To:" section
	
	addrs = addrString.replace(" ","").split(",")

	#remove external emails
	if filterExternal:
		addrs = _filterExternalAddrs(addrs)
			
	#_print("Found addrs: "+str(addrs))
	#input("debug")
	
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

  @filterexternal: If true, exernal emails are disallowed; if "@enron" isn't in the email, it won't be included. This is
  problematic however, as many employees used non Enron addresses in the days of @aol.com.

  return frequencies as a digraph
"""
def BuildStaticGraph(mailDir,filterExternal=True):
	g = Graph()
	g["MyName"] = "Enron static email network"
	emailDict = {}  #index by:
	
	missingEmployees = []
	employeeFolders = [os.path.abspath(os.path.join(mailDir,emp)) for emp in os.listdir(mailDir)]
	_print(employeeFolders)
	#foreach employee, parse all of their sent emails
	for emp in employeeFolders:
    _print("processing "+emp)
		emails = listEmailFiles(emp)
		if len(emails) > 0:
			#probe a few sent emails for the unique sender address
			senderAddr = getSenderEmailId(emails)
			#set up the links
			for email in emails:
				#list the target addresses of this email (there could be more than one)
				targets = listTargetAddresses(email,filterExternal=False)
				if senderAddr not in emailDict.keys():
				  emailDict[senderAddr] = targets
				else: #else, append to existing list
					emailDict[senderAddr] += targets
				#disallow self-links?
			#_print("sender: "+senderAddr)
		else:
			#for reporting: record employees for whom no emails are found
			missingEmployees.append(emp)

	#_print("Emaildict: "+str(emailDict))
	for k in emailDict.keys():
		_print(k+": "+str(emailDict[k]))
		#raw_input("dbg")

	#finally, build the igraph object from the emailDict
	for sender in emailDict.keys():
		for target in emailDict[sender]:
			_addUnweightedLink(sender,target,g)

	return g

#igraph's disgustingly inefficient api for check if node exists, by id. Is there a better way?
def _hasVertex(name,g):
	for node in g.vs:
		_print("name? "+node["name"]+" == "+"name")
		if node["name"] == name:
			_print("true")
			return True
	_print("false")
	return False

#Utility for adding unweighted graph links to g for a sender and receiver
def _addUnweightedLink(src,dest,g):
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
		edgeId = g.get_eid(src, dest, g.is_directed())
		if edgeId == -1:
			g.add_edge(src,dest)
		# else: nothing, graph is undirected, unweighted for now

#enronRootDir = "./maildir"
#_print("sep="+os.sep)
enronRootDir = "./maildir"
g = BuildStaticGraph(enronRootDir,False)
_print(g)
_print("vertices: "+str([v for v in g.vs]))
_print("edges: "+str([e for e in g.es]))
g.write_gml("enronGraph.gml")
#g = Graph.Read("enronGraph.gml")

#g = Graph()
#g.add_vertices(3)
#g.add_edges([(0,1),(1,2),(2,0)])
#g.write_gml("dummy.gml")
#_print(str(g))
