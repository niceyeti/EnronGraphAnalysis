This repo contains some network analysis work on the Enron dataset.

Most scripts assume that the Enron dataset ('maildir') is present within this repo under folder 'maildir', but
I'm not going to push up the actual dataset since it is >1GB. Retrieve the dataset from https://www.cs.cmu.edu/~./enron/
and place it in this directory (./).



buildGraph.py will build a graph based on the enron email dataset given a few parameters.

The Enron email dataset is in the public domain, but contains actual names. Please respect the privacy of
the names/addresses output by this work. In the future we may want to pseudonymize these in some way, out
of respect for privacy.



