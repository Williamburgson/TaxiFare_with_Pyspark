"""
spark-submit flask_example.py
"""
import sys
from random import random
from operator import add
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext#, SparkSession
from pyspark.sql import Row
from collections import OrderedDict
# import spark_model as sm
from flask import (
    Flask, Blueprint, flash, g, redirect, render_template, request, url_for
)
sys.path.append('./python')
app = Flask(__name__)
# spark = SparkSession\
#     .builder\
#     .appName("AppName")\
#     .config("spark.driver.allowMultipleContexts", "true")\
#     .getOrCreate()
# spark.conf.set("spark.sql.shuffle.partitions", 6)
# spark.conf.set("spark.executor.memory", "2g")
# spark.conf.set("spark.driver.allowMultipleContexts", "true")
conf = SparkConf().setAppName('Flask on Spark')\
    .set("spark.driver.allowMultipleContexts", "true")\
    .set("spark.shuffle.service.enabled", "false")\
    .set("spark.dynamicAllocation.enabled", "false")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# file_path = "raw_merged.csv"
file_path = "new_input.csv"

'''
Create linear regression model for prediction
'''
class spark_model:
    def __init__(self, data):
        self.data = data
        lr = LinearRegression(featuresCol='features', labelCol='label',\
                                 maxIter=10, regParam=0.3, elasticNetParam=0.8)
        self.lr_model = lr.fit(data)

    def get(self):
        return self.lr_model


@app.route("/")
def index():
    message = 'hello world'
    return render_template('index.html', message=message)


@app.route("/", methods=['POST'])
def my_form_post():
    # read from form
    origin = request.form['origin']
    destination = request.form['destination']
    passenger = request.form['passengers']
    distance = request.form['distance']

    # initailize params
    prcp = 0.0413
    snow = 0.0603
    tavg = 47.9109
    tmax = 56.8859
    tmin = 39.8287

    # create new df of input data
    def to_row(d):
        return Row(**OrderedDict(sorted(d.items())))
    # df = spark.sparkContext.parallelize([{'PRCP': prcp, 'SNOW': snow, 'TAVG': tavg,\
    #                  'TMAX': tmax, 'TMIN': tmin, 'passenger_count': float(passenger),\
    #                  'trip_distance': float(distance)}]).map(to_row).toDF()
    df = sc.parallelize([{'PRCP': prcp, 'SNOW': snow, 'TAVG': tavg,\
                     'TMAX': tmax, 'TMIN': tmin, 'passenger_count': float(passenger),\
                     'trip_distance': float(distance)}]).map(to_row).toDF()

    #assemble to vector
    flist = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN', 'passenger_count', 'trip_distance']
    data2 = df.select(df.PRCP, df.SNOW, df.TAVG, df.TMAX, df.TMIN, df.passenger_count,
                     df.trip_distance)
    assembler = VectorAssembler(inputCols=flist, outputCol='features')
    temp = assembler.transform(data2)
    input_data = temp.select("features")

    # new model instance
    #train = spark.read.csv(file_path, header=True, inferSchema=True)
    train = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').\
        load("new_input.csv")
    print(train.printSchema())
    train.show(n=5)


    # from pyspark.sql.types import StructType
    # from pyspark.sql.types import StructField
    # from pyspark.sql.types import FloatType, IntegerType, StringType
    #
    # rdd = sc.textFile("new_input.csv")
    # schema = StructType([
    #     StructField("index", StringType(), True),
    #     StructField("PRCP", FloatType(), True),
    #     StructField("SNOW", FloatType(), True),
    #     StructField("TAVG", FloatType(), True),
    #     StructField("TMAX", FloatType(), True),
    #     StructField("TMIN", FloatType(), True),
    #     StructField("passenger_count", FloatType(), True),
    #     StructField("trip_distance", FloatType(), True),
    #     StructField("total_amount", FloatType(), True)
    # ])
    # new_rdd = rdd.map(lambda x: float(x)).map(lambda x: x.split(','))
    # temp = sqlContext.createDataFrame(new_rdd, schema)



    train_data = train.select(train.PRCP, train.SNOW, train.TAVG, train.TMAX, train.TMIN, train.passenger_count,
                           train.trip_distance, train.total_amount.alias('label'))
    train_data = train_data.dropna()
    assembler = VectorAssembler(inputCols=flist, outputCol='features')
    temp = assembler.transform(train_data)
    train_vector = temp.select("features", "label")
    # train_vector = train_vector.cache()
    print(train_vector.printSchema())
    train_vector.show(n=10, truncate=False)
    train_vector.describe().show()


    lr = LinearRegression(featuresCol='features', labelCol='label',
                          maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_vector)
    y_pred = ""#lr_model.transform(data)




    print("=====================\n",lr_model,"\n======================\n",
                        y_pred,"=====================\n\n\n\n\n")

    return str(lr_model.coefficients)



if __name__ == "__main__":
    try:
        app.run(debug=True)
        #app.run(threaded=True)
        #app.run(port=os.environ.get('FLASK_PORT', 8080), host='0.0.0.0')
    finally:
        sc.stop()
        # spark.stop()