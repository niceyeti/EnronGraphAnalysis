#!/bin/bash
#Each command in this file must contain a single space at the end of each line, or bash/python will complain about CR's
sh buildUndirected.sh 
sh buildDirected.sh 
python AnalyzeGraph.py enronUndirected.graphml report_undirected 
python AnalyzeGraph.py enronDirected.graphml report_directed 
echo analysis completed.
