#!/bin/bash
# Builds a whole bunch of test graphs, using as many construction param combinations as possible

#build a basic directed, weighted graph
python BuildGraph.py ./testdir testDirected.graphml --directed --weighted --disallowReflexive --edgeFilter=3 --nodeFilter=3
#build a basic undirected, weighted graph
python BuildGraph.py ./testdir testDirected.graphml --weighted --disallowReflexive --edgeFilter=3 --nodeFilter=3
#build a basic directed, unweighted graph
python BuildGraph.py ./testdir testDirected.graphml --directed --disallowReflexive --edgeFilter=3 --nodeFilter=3
#build a basic undirected, unweighted graph
python BuildGraph.py ./testdir testDirected.graphml --disallowReflexive --edgeFilter=3 --nodeFilter=3
#builds the most complete graph: directed, weighted, reflexive, no filtering
python BuildGraph.py ./testdir testDirected.graphml --directed --weighted --edgeFilter=0 --nodeFilter=0
