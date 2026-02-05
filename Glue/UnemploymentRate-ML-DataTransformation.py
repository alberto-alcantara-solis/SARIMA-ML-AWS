#Automated triggered Glue ETL job (7:35 on 21th of each month) to transform raw unemployment rate data from S3 and store it in DynamoDB for ML processing.

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql.functions import col, array, struct, explode, lit, split, when, concat_ws

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# S3 input path
input_path = "s3://unemployment-rate-ml/raw-data/estat_ei_lmhr_m.tsv"

# Read TSV file
df = spark.read.option("header", True).option("sep", "\t").csv(input_path)

# Rename first column to All
df = df.withColumnRenamed("freq,unit,s_adj,indic,geo\\TIME_PERIOD", "All")

# Split All into separate columns
split_cols = split(col("All"), ",")
df = df.withColumn("S_Adj", split_cols.getItem(2)) \
       .withColumn("Indic", split_cols.getItem(3)) \
       .withColumn("NUTS1", split_cols.getItem(4))  # Rename here directly

# Split Indic into separate columns
split_indic = split(col("Indic"), "-")
df = df.withColumn("Gender", split_indic.getItem(2)) \
       .withColumn("Category", split_indic.getItem(3))  # Rename here directly

# Identify date columns (all columns except All and the split columns)
date_cols = [c for c in df.columns if c not in ["All", "S_Adj", "Indic", "NUTS1", "Gender", "Category"]]

# Melt the date columns into TimePeriod and Value
exprs = array(*[struct(lit(c).alias("TimePeriod"), col(c).alias("Value")) for c in date_cols])
df = df.withColumn("kv", explode(exprs))

# Add ID column as a concatenation of S_Adj, Gender, Category, NUTS1
df = df.withColumn(
    "ID",
    concat_ws("_", "NUTS1", "Gender", "Category", "S_Adj")
)

# Select final columns
df_final = df.select(
    "ID",
    col("kv.TimePeriod").alias("TimePeriod"),
    col("kv.Value").alias("Value"),
    "NUTS1",
    "Gender",
    "Category",
    "S_Adj"
)

# Remove rows where Value is ":"
df_final = df_final.withColumn("Value", when(col("Value").contains(":"), None).otherwise(col("Value")))
df_final = df_final.filter(col("Value").isNotNull())

# Map NUTS1 codes to full country names
nuts_to_country = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "EA20": "Euro area (20 countries)",
    "EA21": "Euro area (21 countries)",
    "EU27_2020": "European Union (27 countries)",
    "BA": "Bosnia and Herzegovina",
    "CH": "Switzerland",
    "EL": "Greece",
    "IS": "Iceland",
    "JP": "Japan",
    "MK": "North Macedonia",
    "NO": "Norway",
    "TR": "Turkey",
    "UK": "United Kingdom",
    "US": "United States"
}


# Add Country column
mapping_expr = when(lit(False), lit(None))  # start with empty expression
for k, v in nuts_to_country.items():
    mapping_expr = mapping_expr.when(col("NUTS1") == k, lit(v))
mapping_expr = mapping_expr.otherwise(col("NUTS1"))  # fallback
df_final = df_final.withColumn("Country", mapping_expr)


# Convert to DynamicFrame for Glue DynamoDB write
from awsglue.dynamicframe import DynamicFrame
dyf = DynamicFrame.fromDF(df_final, glueContext, "dyf")

# DynamoDB output options
dynamo_options = {
    "dynamodb.output.tableName": "UnemploymentRate-ML-DynamoDB-TransformedData",
    "dynamodb.throughput.write.percent": "1.0"
}

# Write to DynamoDB
glueContext.write_dynamic_frame.from_options(
    frame=dyf,
    connection_type="dynamodb",
    connection_options=dynamo_options
)

job.commit()
