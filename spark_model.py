from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark
class spark_model:

    def __init__(self, data):
        conf = pyspark.SparkConf().setAppName('Flask on Spark') \
                .set("spark.driver.allowMultipleContexts", "true")\
                .set("spark.shuffle.service.enabled", "false")\
                .set("spark.dynamicAllocation.enabled", "false")
        sc = pyspark.SparkContext(conf=conf)
        sqlContext = pyspark.SQLContext(sc)

        self.data = data
        lr = LinearRegression(featuresCol='features', labelCol='label',\
                                 maxIter=10, regParam=0.3, elasticNetParam=0.8)

        self.lr_model = lr.fit(data)

    def get(self):
        return self.lr_model