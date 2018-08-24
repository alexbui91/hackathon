from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, OneHotEncoder, StringIndexer
from datetime import timedelta
import numpy as np
import udf as udf_utils

class SparkEngine():

    def __init__(self):
        self.spark = self.init_spark()
        # init udf function
        self.policy_cols = ["policy_id","c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14","c15","c16", \
                    "c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","v00","v01","v02","v03","z00","z01","z02","z03","z04", \
                    "z05","z06","z07","z08","z09"]
        policy_col_t = [StructField(x, IntegerType(), True) for x in self.policy_cols[:19]]  \
                    +  [StructField(x, StringType(), True) for x in self.policy_cols[19:27]] \
                    +  [StructField(x, DoubleType(), True) for x in self.policy_cols[27:33]] \
                    +  [StructField(x, StringType(), True) for x in self.policy_cols[33:]]

        self.policy_schema = StructType(policy_col_t)
        self.policy_cols_norm = self.policy_cols[1:19] + ["v00","v01","v02","v03"]
        # "c12", "c13", "c14", "c15", "c16"
        self.policy_cols_onehot = ["c12"]

        self.result_schema = StructType([StructField("policy_id", IntegerType(), True)])

        self.customer_cols = ["policy_id","c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","v00","v01","v02","v03","v04","v05","v06",\
                        "v07","v08","v09","v10","v11","v12","v13","v14","v15","v16","z00","z01","z02","z03"]

        customer_cols_t = [StructField(x, IntegerType(), True) for x in self.customer_cols[0:8]] \
                        + [StructField(x, StringType(), True) for x in self.customer_cols[8:]]
        self.customer_schema = StructType(customer_cols_t)
        self.customer_cols_norm = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","v01", "v02", "v03","v04","v05","v06","v07","v08","v09", "v10", "v12", "v14"]

        self.claim_cols = ["policy_id","c00","c01","c02","c03","c04","c05","c06","c07","v00","v01","v02","z00","z01","z02","z03","z04","z05","z06"]
        self.claim_cols_norm = ["c00","c01","c03","c04","c05","c06", "v00", "v01", "v02"]
        self.claim_cols_onehot = ["z04"]
        claim_cols_t = [StructField(x, IntegerType(), True) for x in self.claim_cols[0:8]]  \
                    +  [StructField("c07", StringType(), True)] \
                    +  [StructField(x, DoubleType(), True) for x in self.claim_cols[9:12]] \
                    +  [StructField(x, StringType(), True) for x in self.claim_cols[12:]]
        self.claim_schema = StructType(claim_cols_t)
        self.renewal_schema = StructType([StructField("label", IntegerType(), True), StructField("policy_id", IntegerType(), True)])
        self.udf_get_limit = udf(udf_utils.get_limit)
        self.udf_float = udf(udf_utils.fix_float)

    def init_spark(self):
        conf = (SparkConf().setAppName("prediction_airpollution"))
        conf.set("spark.driver.memory", "2g")
        conf.set("spark.executor.memory", "1g")
        conf.set("spark.ui.port", "31040")
        conf.set("spark.sql.shuffle.partitions", "200")
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        return spark

    def normalize_vector(self, df_data, cols):
        vectorAssembler = VectorAssembler(inputCols=cols, outputCol="features_norm")
        mxm = MinMaxScaler(inputCol="features_norm", outputCol="features")
        output = vectorAssembler.transform(df_data)
        mxm_scaler = mxm.fit(output)
        trans = mxm_scaler.transform(output)
        return trans

    def onehot_encode(self, df, features):
        indexers = [StringIndexer(inputCol=f, outputCol=f + "_indexed") for f in features]
        pipeline = Pipeline(stages=indexers)
        indexed = pipeline.fit(df).transform(df)
        
        oh_indexers = [OneHotEncoder(inputCol=f+"_indexed", outputCol=f+"_vector").setDropLast(False) for f in features]
        pipeline.setStages(oh_indexers)
        oh_indexed = pipeline.fit(indexed).transform(indexed)
        
        return oh_indexed

    def getIntegerType(self, schema):
        return [x.name for x in schema.fields if x.dataType == IntegerType()]

    def getDoubleType(self, schema):
        return [x.name for x in schema.fields if x.dataType == DoubleType()]

    def pad_data(self, c_vals, c_dims, m):
        if c_vals:
            c_vals_l = len(c_vals)
            if c_vals_l > m:
                c_vals = c_vals[:m]
                c_vals_ = [float(x) for c in c_vals for x in c]
                c_vals_ = np.reshape(c_vals_, (m, c_dims))
            else:
                c_vals_ = [float(x) for c in c_vals for x in c]
                c_vals_ = np.pad(c_vals_, (0, (m * c_dims - len(c_vals_))), 'constant', constant_values=0)
                c_vals_ = np.reshape(c_vals_, (m, c_dims))
        else:
            c_vals_ = np.zeros((m, c_dims))
        return c_vals_

    def get_data_from_df(self, policies):
        policy_vectors = []
        claim_vectors = []
        customer_vectors = []
        labels = []
        m = 5
        c_dims = 22
        cl_dims = 15
        for p in policies:
            p_v = [float(x) for x in p["all_features"]]
            c_vals = p["cus_features"]
            c_vals_ = self.pad_data(c_vals, c_dims, m)
            cl_vals = p["cl_features"]
            cl_vals_ = self.pad_data(cl_vals, cl_dims, m)
            if "label" in p:
                labels.append(int(p["label"]))
            customer_vectors.append(c_vals_)
            claim_vectors.append(cl_vals_)
            policy_vectors.append(p_v)
        return policy_vectors, claim_vectors, customer_vectors, labels

    def process_vectors(self):
        # engine do something
        p1 = "release/claim.csv"
        p2 = "release/customer.csv"
        p3 = "release/policy.csv"
        p4 = "release/renewal_sample.csv"
        p5 = "release/result.csv"
        
        results = self.spark.read.format("csv").option("header", "true").schema(self.result_schema).load(p5)
        claim = self.spark.read.format("csv").option("header", "true").schema(self.claim_schema).load(p1)\
                    .na.fill(0.0, self.claim_cols[9:12])                   
        claim_norm = self.normalize_vector(claim, self.claim_cols_norm)
        claim_norm = self.onehot_encode(claim_norm, self.claim_cols_onehot)
        claim_assem = VectorAssembler(inputCols=["features"] + [x + "_vector" for x in self.claim_cols_onehot], outputCol="all_features")
        claim_norm = claim_assem.transform(claim_norm).select(col("policy_id"), col("z03"), col("all_features").alias("cl_features"))
        
        customer = self.spark.read.format("csv").option("header", "true").schema(self.customer_schema).load(p2)\
                    .withColumn("c07", self.udf_float(col("c07")).cast("double"))\
                    .withColumn("c08", self.udf_float(col("c08")).cast("double"))\
                    .withColumn("c09", self.udf_float(col("c09")).cast("double"))\
                    .withColumn("v00", self.udf_float(col("v00")).cast("double"))\
                    .withColumn("v01", self.udf_float(col("v01")).cast("double"))\
                    .withColumn("v02", self.udf_float(col("v02")).cast("double"))\
                    .withColumn("v03", self.udf_float(col("v03")).cast("double"))\
                    .withColumn("v04", self.udf_float(col("v04")).cast("double"))\
                    .withColumn("v05", self.udf_float(col("v05")).cast("double"))\
                    .withColumn("v06", self.udf_float(col("v06")).cast("double"))\
                    .withColumn("v07", self.udf_float(col("v07")).cast("double"))\
                    .withColumn("v08", self.udf_float(col("v08")).cast("double"))\
                    .withColumn("v09", self.udf_float(col("v09")).cast("double"))\
                    .withColumn("v10", self.udf_float(col("v10")).cast("double"))\
                    .withColumn("v11", self.udf_float(col("v11")).cast("double"))\
                    .withColumn("v12", self.udf_float(col("v12")).cast("double"))\
                    .withColumn("v13", self.udf_float(col("v13")).cast("double"))\
                    .withColumn("v14", self.udf_float(col("v14")).cast("double"))\
                    .withColumn("v15", self.udf_float(col("v15")).cast("double"))\
                    .withColumn("v16", self.udf_float(col("v16")).cast("double"))\
                    .na.fill(0, self.getIntegerType(self.customer_schema))
        customer_norm = self.normalize_vector(customer, self.customer_cols_norm).select(col("policy_id"), col("z01"), col("features").alias("cus_features"))
        
        pint = self.getIntegerType(self.policy_schema)
        del pint[0]
        policy = self.spark.read.format("csv").option("header", "true").schema(self.policy_schema).load(p3)\
                    .filter(col("policy_id").isNotNull()) \
                    .na.fill(0, pint)\
                    .na.fill(0.0, self.getDoubleType(self.policy_schema))
        
        policy_norm = self.normalize_vector(policy, self.policy_cols_norm)
        #policy_norm = self.onehot_encode(policy_norm, self.policy_cols_onehot)
        
        policy_assem = VectorAssembler(inputCols=["features"] + [x + "_vector" for x in self.policy_cols_onehot], outputCol="all_features")    
        policy_norm = policy_assem.transform(policy_norm)

        renewal = self.spark.read.format("csv").option("header", "true").schema(self.renewal_schema).load(p4).alias("renewal")

        claim_ = claim_norm.groupBy(col("policy_id")).agg(collect_list("cl_features").alias("cl_features")).select("policy_id", "cl_features")
        customer_ = customer_norm.groupBy(col("policy_id")).agg(collect_list("cus_features").alias("cus_features")).select("policy_id", "cus_features")
        
        policy_df = policy_norm.join(renewal, [policy.policy_id == renewal.policy_id])\
                    .join(claim_, [policy.policy_id == claim_.policy_id], "left_outer")\
                    .join(customer_, [policy.policy_id == customer_.policy_id], "left_outer")\
                    .select("renewal.*", policy_norm.all_features, claim_.cl_features, customer_.cus_features)

        test_policy_df = policy_norm.join(results, [policy.policy_id == results.policy_id])\
                    .join(claim_, [policy.policy_id == claim_.policy_id], "left_outer")\
                    .join(customer_, [policy.policy_id == customer_.policy_id], "left_outer")\
                    .select(results.policy_id, policy_norm.all_features, claim_.cl_features, customer_.cus_features)

        policies = policy_df.collect()
        test_policies = test_policy_df.collect()

        train_data = self.get_data_from_df(policies)
        test_data = self.get_data_from_df(test_policies)
    
        return train_data, test_data