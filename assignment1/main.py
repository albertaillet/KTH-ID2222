# %% Imports
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from nltk.corpus import reuters

# %% Create Spark Session
spark = SparkSession.builder.master('local[*]').appName("Assignment 1").getOrCreate()
sc = spark.sparkContext

# %% Load all documents
documents = reuters.fileids()
text = [reuters.raw(doc) for doc in documents]

# %% Create a dataframe with the documents
pandas_df = pd.DataFrame({'document': documents, 'text': text})

# %%
df = spark.createDataFrame(pandas_df)

# %%
k = 10
@F.udf(returnType=ArrayType(StringType()))
def shingle(text: str):
    return list(set(hash(text[i : i + k]) for i in range(len(text) - k + 1)))

# %%
df = df.withColumn('shingles', shingle('text'))

# %%
udf_len = F.udf(len, IntegerType())
df = df.withColumn('n_shingles', udf_len('shingles'))

# %%
udf_count = F.udf(len, IntegerType())
df = df.withColumn('text_length', udf_count('text'))

# %%
class Jaccard:
    @staticmethod
    @F.udf(returnType=FloatType())
    def distance(list1, list2):
        set1, set2 = set(list1), set(list2)
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union)


# %%
df = df.withColumn('jaccard', Jaccard.distance('shingles', 'shingles'))

# %%
