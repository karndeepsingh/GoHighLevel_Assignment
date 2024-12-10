from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lower, regexp_replace, to_date, when, split
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, FeatureHasher, VectorAssembler

def initialize_spark(config):
    """
    Initialize a Spark session based on the configuration.
    """
    spark = SparkSession.builder \
        .appName(config["spark"]["app_name"]) \
        .config("spark.executor.memory", config["spark"]["executor_memory"]) \
        .config("spark.driver.memory", config["spark"]["driver_memory"]) \
        .config("spark.executor.cores", config["spark"]["executor_cores"]) \
        .config("spark.num.executors", config["spark"]["num_executors"]) \
        .getOrCreate()
    return spark

def preprocess_data(spark, data_path):
    """
    Load and preprocess the dataset.
    """
    # Load data
    books_df = spark.read.csv(
        data_path,
        header=True,
        inferSchema=True,
        quote='"',
        escape='"',
        multiLine=True
    )

    # Drop unnecessary columns
    books_df = books_df.drop("_c0")

    # Fill missing values
    books_df = books_df.fillna({
        "description": "Unknown",
        "Title": "Unknown",
        "categories": "Unknown",
        "publisher": "Unknown"
    })

    # Merge text columns
    books_df = books_df.withColumn("merged_text", concat_ws(" ", col("Title"), col("description")))
    books_df = books_df.withColumn("merged_text", lower(col("merged_text")))
    books_df = books_df.withColumn("merged_text", regexp_replace(col("merged_text"), "[^a-zA-Z\\s]", ""))

    # Text preprocessing
    tokenizer = Tokenizer(inputCol="merged_text", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=500)
    idf = IDF(inputCol="rawFeatures", outputCol="tfidf_features")

    # Apply transformations
    books_df = tokenizer.transform(books_df)
    books_df = stopwords_remover.transform(books_df)
    books_df = hashing_tf.transform(books_df)
    idf_model = idf.fit(books_df)
    books_df = idf_model.transform(books_df)

    # Categorical feature hashing
    hasher = FeatureHasher(
        inputCols=["categories", "authors", "publisher"],
        outputCol="hashed_features",
        numFeatures=500
    )
    books_df = hasher.transform(books_df)

    # Assemble all features
    assembler = VectorAssembler(
        inputCols=["tfidf_features", "hashed_features"],
        outputCol="features"
    )
    books_df = assembler.transform(books_df)

    return books_df
