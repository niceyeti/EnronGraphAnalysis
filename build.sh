#!/bin/bash
# Builds a weighted, directed graph, disallowing reflexive loops (self-emails), filtering any edges between peers sharing fewer than 3 emails
python BuildGraph.py ./maildir enronGraph.gml --directed --weighted --disallowReflexive --frequencyFilter=3

