#! /bin/bash

# Remove old "time log"
rm ./logs/time.txt

# Spark-submit
spark-submit --master local[20] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 1 --master "local[20]"
spark-submit --master local[20] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 10 --master "local[20]"
spark-submit --master local[20] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 100 --master "local[20]"
spark-submit --master local[20] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 1000 --master "local[20]"

# Rename as "local 20"
mv "./logs/time.txt" "./logs/time_20.txt"

# Spark-submit
spark-submit --master local[1] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 1 --master "local[1]"
spark-submit --master local[1] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 10 --master "local[1]"
spark-submit --master local[1] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 100 --master "local[1]"
spark-submit --master local[1] --executor-memory 100G --driver-memory 100G ClassifyEdges.py --N 1000 --master "local[1]"

# Rename as "local 01"
mv "./logs/time.txt" "./logs/time_01.txt"