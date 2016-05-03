#!/bin/bash
# Builds a weighted, directed graph, disallowing reflexive loops (self-emails), filtering any edges between peers sharing fewer than 30 emails
python BuildGraph.py ./maildir enronDirected.graphml --directed --weighted --disallowReflexive --edgeFilter=8 --nodeFilter=8
