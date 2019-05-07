import sys
from random import random
from operator import add
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row
from collections import OrderedDict

conf = SparkConf().setAppName('Flask on Spark')\
    .set("spark.driver.allowMultipleContexts", "true")\
    .set("spark.shuffle.service.enabled", "false")\
    .set("spark.dynamicAllocation.enabled", "false")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def toVector(df):
    return None

def read_csv(filename):
    df = sqlContext.read.format('com.databricks.spark.csv').\
        options(header='true', inferSchema='true'). \
        load(filename)
    print("\n\n\n", df.printSchema(), "\n\n\n\n")
    print("\n\n\n", df.show(n=5), "\n\n\n\n")
    return df

def train_model(input_vector):
    lr = LinearRegression(featuresCol='features', labelCol='label',
                          maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(input_vector)
    print("\n\nCoefs\n", lr_model.coefficients, "\n\n\n\n")
    return lr_model

def predict(input_vector, lr):
    return None

def to_row(d):
    return Row(**OrderedDict(sorted(d.items())))

if __name__ == "__main__":
    # read the training csv
    train = read_csv("new_input.csv")

    # set up assembler
    flist = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN', 'passenger_count', 'trip_distance'] # feature list
    assembler = VectorAssembler(inputCols=flist, outputCol='features')

    # transform training df to vector
    train_data = train.select(train.PRCP, train.SNOW, train.TAVG, train.TMAX, train.TMIN, train.passenger_count,
                              train.trip_distance, train.total_amount.alias('label'))
    train_data = train_data.dropna()
    temp = assembler.transform(train_data)
    train_vector = temp.select("features", "label")
    print(train_vector.show(n=10, truncate=False))

    # train model with train vector
    lr_model = train_model(train_vector)

    # initialize test data
    prcp = 0.0413
    snow = 0.0603
    tavg = 47.9109
    tmax = 56.8859
    tmin = 39.8287
    passenger = 3
    distance = 10

    # transform test data to vector
    test_df = sc.parallelize([{'PRCP': prcp, 'SNOW': snow, 'TAVG': tavg,\
                     'TMAX': tmax, 'TMIN': tmin, 'passenger_count': float(passenger),\
                     'trip_distance': float(distance)}]).map(to_row).toDF()
    test_data = test_df.select(test_df.PRCP, test_df.SNOW, test_df.TAVG, test_df.TMAX,
                               test_df.TMIN, test_df.passenger_count, test_df.trip_distance)
    temp = assembler.transform(test_data)
    test_vector = temp.select("features")

    # use trained model to predict
    y_predict = predict(test_vector, lr_model)
    print(y_predict)

