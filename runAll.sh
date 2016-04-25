#!/bin/bash

#sh buildUndirected.sh
#sh buildDirected.sh
python AnalyzeGraph.py enronUndirected.graphml report_undirected 
python AnalyzeGraph.py enronDirected.graphml report_directed 
echo analysis completed.
