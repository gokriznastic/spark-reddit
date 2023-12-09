
# README: Spark-Reddit Project

EECE 5645 - Final Project

This README file contains information on the Parallel Processing Final Project: Social Network Analysis for Reddit. This project consists of the following:

 - ClassifyEdges.py file
 - EvaluateTime.sh
 - data folder
 - logs folder
 - save_models folder
 - save_preds folder

# Python File

All python files are individually commented and documented. For more information on a particular function within a python file, please import the module into a python environment and use the "_help(FuncName)_".

## ClassifyEdges.py

This python file requires the following modules to be installed in the python environment:
 - numpy
 - argparse
 - sys
 - os
 - shutil
 - time
 - findspark
 - pyspark
 - pdb

For information on all positional arguments use the following: "_python ClassifyEdges.py -h_" or "_python ClassifyEdges.py -help_".

To execute this file, use spark-submit. For example: "_spark-submit --master local[20] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 1000_"

# Bash Script

## EvaluateTime.sh

This script compares the performance of a non-parallelized execution to a parallel execution using a single local thread "local[1]" as well as 20 local threads "local[20]". The resulting "time output" files are saved under a new name, for later use.

# Folders

## data folder

The data folder contains all the edge data of the graph. It contains two files:

- soc-redditHyperlinks-body.tst
-- Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the body of the post
- soc-redditHyperlinks-title.tsv
-- Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the title of the post.

Each datapoint is formatted in the following way: [SOURCE_SUBREDDIT, TARGET_SUBREDDIT, POST_ID, TIMESTAMP, POST_LABEL, POST_PROPERTIES]

## logs folder

This folder contains two log files generated by the ClassifyEdges code:

- output.txt
-- Contains information about the output of the most recent run. This output file contains most of the metrics and analysis results of the model.
- time.txt
-- Contains information about the execution time for a particular run.

## models folder

This folder contains the linear regression model. They are saved with automatically generated names that depend on the following: the regularization parameter, the elastic parameter, and the number of max-iterations N.

## preds folder

This folder contains the model prediction results. They are saved with automatically generated names that depend on the following: the regularization parameter, the elastic parameter, and the number of max-iterations N.


