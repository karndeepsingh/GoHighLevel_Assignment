import time
from pyspark.sql.functions import col, abs
from model_training import train_model 




def calculate_mape(predictions, label_col="Impact", prediction_col="prediction"):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    # Add a new column for absolute percentage error
    mape_df = predictions.withColumn(
        "abs_error", abs((col(label_col) - col(prediction_col)) / col(label_col))
    )
    # Calculate average MAPE
    mape = mape_df.agg({"abs_error": "avg"}).collect()[0][0] * 100  # Convert to percentage
    return mape




def evaluate_training_time(spark, train_data, test_data,model, config):
    """
    Evaluate total training time for different worker configurations.
    """
    worker_configs = config["evaluation"]["training_time_evaluation"]["worker_configs"]
    training_times = {}

    print("Evaluating training time for different worker configurations...")
    for workers in worker_configs:
        print(f"Setting up {workers} worker(s)...")
        
        spark.conf.set("spark.executor.instances", workers)

        # Measure training time
        start_time = time.time()
        best_model, predictions, rmse, mape = train_model(train_data, test_data, model, config)
        
        end_time = time.time()

        elapsed_time = end_time - start_time
        training_times[workers] = elapsed_time
        print(f"Training with {workers} worker(s) completed in {elapsed_time:.2f} seconds.")

    print("Training Time Results:")
    for workers, time_taken in training_times.items():
        print(f"{workers} Worker(s): {time_taken:.2f} seconds")
