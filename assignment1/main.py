# %% Imports
import pandas as pd
from pyspark.sql import SparkSession
from nltk.corpus import reuters

# %% Create Spark Session
spark = SparkSession.builder.master('local[*]').appName("Assignment 1").getOrCreate()
sc = spark.sparkContext

# %% Load all documents
documents = reuters.fileids()
text = [reuters.raw(doc) for doc in documents]

# %% Create a dataframe with the documents
df = pd.DataFrame({'document': documents, 'text': text})

# %%
df = spark.createDataFrame(df)