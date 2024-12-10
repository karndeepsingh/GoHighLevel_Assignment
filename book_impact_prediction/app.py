from data_preprocessing import initialize_spark, preprocess_data
from utils.helper_functions import load_config
from model_training import train_model
from evaluation import evaluate_training_time
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, DecisionTreeRegressor


def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Initialize Spark
    spark = initialize_spark(config)
    
    # Preprocess data
    books_df = preprocess_data(spark,  config["data_path"])
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    # Train and evaluate model
    ## Checking if the evaluation different spark worker instances.
    if config["evaluation"]["training_time_evaluation"]["enable"]:
        evaluate_training_time(spark, train_data, test_data, config)
    else:
        # Proceed with standard model training
        print("Training time evaluation is disabled. Proceeding with model training...")
        model = LinearRegression(featuresCol="features", labelCol="Impact")
        best_model, predictions, rmse, mape = train_model(train_data, test_data, model, config)
        
        # Print evaluation metrics
        print(f"RMSE: {rmse}")
        print(f"MAPE: {mape}")

if __name__ == "__main__":
    main()
