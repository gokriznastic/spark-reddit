"""
authors: Laura Rojas, Ahmet Demirkaya, Gopal Krishna, Beyza Cavdar
summary: The ClassifyEdges.py file contains the code necessary to train a LogisticRegression model to classify an graph edge as "positive or negative" based on certain features.
"""
# Import basic libraries
import numpy as np
import argparse
# Import operating system utilities
import sys
import os 
import shutil
from time import time
# Import Spark libraries
import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, split, when
# Import Spark MLlib libraries
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import  BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
# Import python debugger
import pdb


def print_confusion_matrix(predictions):
    """
    authors: Ahmet Demirkaya
    summary: Function to print confusion matrix
    """
    # Convert DataFrame to RDD
    predictionAndLabels = predictions.select(['prediction', 'LINK_SENTIMENT']).rdd.map(lambda x: (float(x[0]), float(x[1])))
    # Instantiate MulticlassMetrics
    metrics = MulticlassMetrics(predictionAndLabels)
    # Print the confusion matrix
    print("Confusion Matrix:\n", metrics.confusionMatrix().toArray())


def printDatasetStats(df):
    """
    authors: Gopal Krishna
    summary: Print basic statistics about the dataset.
    """
    print("Dataset Statistics:")
    print(f"Number of Rows: {df.count()}")
    print(f"Number of Columns: {len(df.columns)}")
    print("Column Names: ", df.columns)
    df.describe().show()


def getData(datafolder, spark, pct=1.0):
    """
    authors: Laura Rojas, Gopal Krishna
    summary: Read dataset from given datafolder and calculate its statistics.
    """
    # Declare "user defined functions"
    floatArray = udf(lambda x: parseParams(x), ArrayType(DoubleType()) )
    intCast = udf(lambda x: int(x), IntegerType() )

    # First, pull graph data from "data" folder (will take ALL FILES in the folder)
    df = spark.read.options(header='True',delimiter='\t').csv(datafolder)

    # Sample the data if pct is less than 1.0
    if pct < 1.0:
        df = df.sample(withReplacement=False, fraction=pct, seed=42)

    # Apply user-defined functions
    df = df.withColumn("LINK_SENTIMENT", intCast(col("LINK_SENTIMENT")))
    df = df.withColumn("PROPERTIES", floatArray(col("PROPERTIES")))

    # Make link sentiment nonnegative
    df = df.withColumn("LINK_SENTIMENT", when(col("LINK_SENTIMENT") == -1, 0).otherwise(1))

    # Print dataset statistics
    printDatasetStats(df)

    # Return dataframe
    return df


def parseParams(s):
    """
    authors: Laura Rojas
    summary: This function parses the parameters in "comma separated string" format
    """
    v = s.split(",")
    x = [0]*len(v)
    for r in range(0,len(v)):
       x[r] = float(v[r])
    return x


def getVectorized(data, split):
    """
    authors: Laura Rojas
    summary: This function vectorizes the raw data in the "PROPERTY" column and returns a train/test split of this data.
    """
    # Obtain temporary "select properties" dataframe and corresponding columns
    select_properties = data.select('POST_ID','LINK_SENTIMENT', *[col('PROPERTIES').getItem(i).alias("PROPERTY"+str(i+1)) for i in range(0, 86)])
    property_columns = ["PROPERTY"+str(i+1) for i in range(0, 86)]

    # Define and apply the "vector assembler". Then, keep only the post id, link sentiment and features columns
    assembler = VectorAssembler(inputCols=property_columns, outputCol="FEATURES")
    vectorized = assembler.transform(select_properties)
    vectorized = vectorized.select(['POST_ID','LINK_SENTIMENT','FEATURES'])
    
    # Apply train-test split and return
    train, test = vectorized.randomSplit(split, seed = 2018)
    # Cache data (will be used later)
    train.cache()
    test.cache()
    
    # Return vectorized data
    return train, test

def getNormalized(train, test):
    """
    authors: Beyza Cavdar
    summary: This function vectorizes the raw data in the "PROPERTY" column and returns the vectorized form.
    """
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(inputCol="FEATURES", outputCol="FEATURES_NORMALIZED")
    scalerModel = scaler.fit(train)

    # Scale the train and test data
    newtrain = scalerModel.transform(train)
    newtest = scalerModel.transform(test)
    
    # Cache/unpersist train and test
    newtrain.cache()
    newtest.cache()
    train.unpersist()
    test.unpersist()

    # Return new train and test
    return newtrain, newtest

def initTime(args):
    """
    authors: Laura Rojas, Gopal Krishna
    summary: A function that initializes the start time and begins storing information in the "time output file".
    """
    # Obtain start time
    start = time()
    # Define "save string"
    savestr = "\nStoring time data for the following parameters: --master " + str(args.master) + \
            " --N " + str(args.N) + \
            " --regParam " + str(args.regParam) + \
            " --elasticParam " + str(args.elasticParam) + \
            " --pct " + str(args.pct)
        # Initialize time data
    file = open('./logs/time.txt','a')
    file.write(savestr)
    file.close()
    # Return start time
    return start

def getTime(start,savestr):
    """
    authors: Laura Rojas
    summary: A function that obtains the amount of time spent training in hours, minutes and seconds. It writes this information in a log file.
    """
    # Get current time for print
    now = time()-start
    h = int(now/3600)
    m = int(now/60 - 60*h)
    s = int(now -60*m -3600*h)
    # Get current time string
    currenttime = str(h) + "h " + str(m)+"m "  + str(s) + "s"
    # Store time data
    file = open('./logs/time.txt','a')
    file.write("\n"+savestr+currenttime)
    file.close()
    # Return current time string
    return currenttime
    
def trainLoop(train, test, args):
    """
    authors: Ahmet Demirkaya, Laura Rojas
    summary: Training loop function that trains the logistic regression regression model.
    """
    # Initialize balance to true
    balance = True
    if balance:
        # Calculate balance ratio (assuming 0 is the minority class)
        balance_ratio = train.filter(col("LINK_SENTIMENT") == 1).count() / train.count()

        # Add a weight column to the DataFrame
        newtrain = train.withColumn("classWeight", when(train["LINK_SENTIMENT"] == 0, balance_ratio).otherwise(1-balance_ratio))
        # Swap oldtrain with newtrain (because of balance). Manage cache and unpersist. 
        newtrain.cache()
        train.unpersist()
        train = newtrain

        # Now use this weight column in logistic regression
        lr = LogisticRegression(featuresCol='FEATURES_NORMALIZED', labelCol='LINK_SENTIMENT', weightCol="classWeight",  maxIter=args.N, regParam=args.regParam, elasticNetParam=args.elasticParam)
    else:
        # Define "logistic regression" model and fit with train data
        lr = LogisticRegression(featuresCol = 'FEATURES_NORMALIZED', labelCol = 'LINK_SENTIMENT', maxIter=args.N, regParam=args.regParam, elasticNetParam=args.elasticParam)
    
    # Fit to training data and cache model
    lr = lr.fit(train)
    
    # Return lr
    return lr, train
    
def getMetrics(lr, test):
    """
    authors: Ahmet Demirkaya, Laura Rojas
    summary: Function that prints the desired "model result metrics".
    """
    # Access the summary
    trainingSummary = lr.summary
    # Print the objective history (loss at each iteration)
    print("Objective History: ")
    print(trainingSummary)
    
    # Make predictions on test data
    predictions = lr.transform(test)
    # Select example rows to display
    predictions.select("prediction", "LINK_SENTIMENT", "FEATURES_NORMALIZED").show(25)
    # Generate predictions for training set
    train_predictions = lr.transform(train)
    # Cache the test and train predictions
    predictions.cache()
    train_predictions.cache()
    
    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(labelCol="LINK_SENTIMENT", rawPredictionCol="prediction", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(predictions)
    print("Test Area Under ROC: " + str(roc_auc))
    
    # Evaluate accuracy on the training dataset
    train_accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="LINK_SENTIMENT", predictionCol="prediction", metricName="accuracy")
    train_accuracy = train_accuracy_evaluator.evaluate(train_predictions)
    print("Training Accuracy: " + str(train_accuracy))
    # Evaluate accuracy on the test dataset
    test_accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="LINK_SENTIMENT", predictionCol="prediction", metricName="accuracy")
    test_accuracy = test_accuracy_evaluator.evaluate(predictions)
    print("Test Accuracy: " + str(test_accuracy))
    
    # Printing confusion matrix for the training set
    print("Training Confusion Matrix:")
    print_confusion_matrix(train_predictions)
    # Printing confusion matrix for the test set
    print("\nTest Confusion Matrix:")
    print_confusion_matrix(predictions)
    
    # Unpersist the train_predictions, no longer needed. Keep test predictions cached (will unpersist in saveData)
    train_predictions.unpersist()
    
    # Return ONLY the predictions (needed to save)
    return predictions

def saveData(lr, predictions, args):
    """
    authors: Ahmet Demirkaya
    summary: Function that saves the model and predictions in the corresponding files.
    """
    # Saving the model
    model_path = "./models/model_reg{}_elast{}_N{}".format(str(args.regParam), str(args.elasticParam), str(args.N))  # Replace with your desired path
    predictions_path = "./preds/model_reg{}_elast{}_N{}".format(str(args.regParam), str(args.elasticParam), str(args.N))  # Replace with your desired path
    # Check if the model path exists and delete it if it does
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    # Save the model
    lr.save(model_path)
    # Print info
    print("Model saved to "+str(model_path))

    # Save the predictions
    if os.path.exists(predictions_path):
        shutil.rmtree(predictions_path)
    # Save the predictions
    predictions.write.format("parquet").save(predictions_path)
    # Print info
    print("Predictions saved to "+str(predictions_path))

    # Unpersist the predictions rdd after saving
    predictions.unpersist()

    # Empty return, all data saved
    return

if __name__ == "__main__":
    """
    authors: Laura Rojas, Ahmet Demirkaya, Gopal Krishna, Beyza Cavdar
    summary: The main function only runs if the module is run directly as a python executable. It proves the functionality of the "classify edges" project
    data: https://snap.stanford.edu/data/soc-RedditHyperlinks.html
    """
    # Declare parser
    parser = argparse.ArgumentParser(description = 'Graph Edge Classification using Parallel Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--master',default="local[*]",help="Spark Master")
    parser.add_argument('--datafolder', default='data', help='Folder where the data is pulled from when training')
    parser.add_argument('--pct', type=float, default=1.0, help='Percentage of dataset to sample')
    parser.add_argument('--regParam', type=float, default=0.1, help='Regularization parameter for LR when training')
    parser.add_argument('--elasticParam', type=float, default=0.8, help='Elastic Net parameter for LR ')
    parser.add_argument('--N', type=int, default=10000000, help='Number of iterations for LR ')
    parser.add_argument('--loglevel', type=str, default="ERROR", help='Spark log level. Default is ERROR only')
    # Define args
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    
    ###
    # INITIALIZATION
    ###
    
    # Define spark context. Declare log level. Default master is "run locally with as many threads as # of processors available locally"
    sc = SparkContext(args.master, 'Graph Edge Classification using Parallel Logistic Regression')
    sc.setLogLevel(args.loglevel)
    print("\nSetting LogLevel to "+ str(args.loglevel))
    # Define sparksession (default master is "run locally with as many threads as # of processors available locally"
    spark = SparkSession.builder.appName("DataFrame").getOrCreate() 
    
    # Initialize start time
    start = initTime(args)
    
    ###
    # PREPROCESSING
    ###
    
    # Obtain raw data 
    rawdata = getData(args.datafolder, spark, args.pct)
    print("Raw data obtained from "+ str(args.datafolder))
    
    # From the raw data, obtain vectorized 
    train, test = getVectorized(rawdata,[0.7, 0.3])
    print("Data vectorized. Train/Test split complete: [0.7, 0.3]")
    
    # From the vectorized data, obtain normalized. Note: train and test will be CACHED
    train, test = getNormalized(train, test)
    print("Data normalized according to training data.")
    # Print preprocess time
    print("Preprocessing time:", getTime(start,"    preprocessing time: "))
    preprocess_time = time()
    
    ###
    # TRAINING
    ###
    
    # Training loop
    print("\nBegin training.")
    lr, train = trainLoop(train, test, args)
    print("Training done.\nTraining time:", getTime(preprocess_time,"    training time: "))
    
    ###
    # RESULTS / METRICS
    ###
    
    # Redirect to OUTPUT log file
    print("\nGenerating metrics. Saving data to output log.")
    # Write to log
    original = sys.stdout
    sys.stdout = open('./logs/output.txt','wt')
    # Get metrics (empty return, all data saved to output log file"
    predictions = getMetrics(lr, test)
    
    # Unpersist train and test (will no longer be used)
    train.unpersist()
    test.unpersist()
    
    ###
    # SAVING RESULTS
    ###
    
    # Redirect from output log file to original (console) output
    print("\nMetrics saved. Saving model and predictions.")
    # Return to console output
    sys.stdout = original
    
    # Save model and predictions using saveData
    saveData(lr, predictions, args)
    
    # Print total execution time
    print("\nTotal execution time:", getTime(start,"    total execution time: "))
    