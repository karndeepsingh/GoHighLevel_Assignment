import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from model_training import train_model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from utils.evaluation import calculate_mape
from pyspark.ml.evaluation import RegressionEvaluator

def train_model(train_data, test_data, model, config,label_col="Impact", prediction_col="prediction"):
    """
    Train the model, perform cross-validation, and evaluate using RMSE and MAPE.
    """
    with mlflow.start_run():
    # Create pipeline with the model
        pipeline = Pipeline(stages=[model])
        param_grid = ParamGridBuilder().build()

        # Set up CrossValidator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse"),
            numFolds=2
        )

        # Fit model
        cv_model = cv.fit(train_data)

        # Make predictions
        predictions = cv_model.bestModel.transform(test_data)

        # Evaluate predictions
        for metric in config["evaluation"]["metrics"]:
            if metric == "RMSE":
                rmse=evaluator.evaluate(predictions)
                print(f"RMSE: {rmse}")
            elif metric == "MAPE":
                mape = calculate_mape(predictions, label_col, prediction_col)
                print(f"MAPE: {mape}")
    

        # Log metrics to MLflow
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)

        # Log model to MLflow
        mlflow.spark.log_model(cv_model.bestModel, "best_model")

        return cv_model.bestModel, predictions, rmse, mape

