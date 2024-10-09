import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
import xgboost as xgb
from hyperopt.pyll import scope
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
from datetime import datetime
from mlflow import MlflowClient
import mlflow.pyfunc
from mlflow.entities import ViewType


@task(name="Read data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(name="Hyper-Parameter Tuning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv):
    
    mlflow.xgboost.autolog()
    
    training_dataset = mlflow.data.from_numpy(X_train.data, targets=y_train, name="green_tripdata_2024-01")
    
    validation_dataset = mlflow.data.from_numpy(X_val.data, targets=y_val, name="green_tripdata_2024-02")
    
    train = xgb.DMatrix(X_train, label=y_train)
    
    valid = xgb.DMatrix(X_val, label=y_val)
    
    def objective(params):
        with mlflow.start_run(nested=True):
             
            # Tag model
            mlflow.set_tag("model_family", "xgboost")
            
            # Train model
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=10
            )
            
            # Predict in the val dataset
            y_pred = booster.predict(valid)
            
            # Calculate metric
            rmse = root_mean_squared_error(y_val, y_pred)
            
            # Log performance metric
            mlflow.log_metric("rmse", rmse)
    
        return {'loss': rmse, 'status': STATUS_OK}

    with mlflow.start_run(run_name="Xgboost Hyper-parameter Optimization", nested=True):
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'objective': 'reg:squarederror',
            'seed': 42
        }
        
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["seed"] = 42
        best_params["objective"] = "reg:squarederror"
        
        mlflow.log_params(best_params)

    return best_params


@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best model ever"):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(best_params)

        # Log a fit model instance
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    
    return None


@task(name="Model Registry")
def model_registry(model_name: str, model_version: str = "1", alias: str = "champion"):
    """Register and transition the model version in the MLflow registry."""
    
    # MLflow client initialization
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        # Get the experiment by name
    experiment = client.get_experiment_by_name(model_name)

    if experiment is None:
        raise ValueError(f"Experiment with name {model_name} not found")
    
    experiment_id = experiment.experiment_id
    
    # Search for all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id], 
                          filter_string="metrics.rmse IS NOT NULL",
                          run_view_type=ViewType.ACTIVE_ONLY,
                          order_by=["metrics.rmse ASC"])


    # Check if any runs were found
    if runs.empty:
        raise ValueError(f"No runs found for experiment {model_name}")

    # Get the run with the lowest RMSE (first run, since it's ordered)
    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_rmse = best_run["metrics.rmse"]

    # Define model URI based on the run_id
    run_uri = f"runs:/{best_run_id}/model"
    
    # Register the model
    result = mlflow.register_model(
        model_uri=run_uri,
        name=model_name
    )

    # Update the model registry with a description
    client.update_registered_model(
        name=model_name,
        description=f"Model registry for {model_name} project",
    )
    
    # Transition the registered model version to the alias (e.g., "champion")
    date = datetime.today()

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version
    )

    client.update_model_version(
        name=model_name,
        version=model_version,
        description=f"The model version {model_version} was transitioned to {alias} on {date}",
    )

    # Load the champion version of the model
    model_uri = f"models:/{model_name}@{alias}"
    champion_version = mlflow.pyfunc.load_model(
        model_uri=model_uri
    )
    
    return champion_version


@flow(name="Main Flow")
def main_flow(year: str, month_train: str, month_val: str) -> None:
    """The main training pipeline"""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    # MLflow settings
    dagshub.init(url="https://dagshub.com/colome8/nyc-taxi-time-prediction.mlflow", mlflow=True)
    
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # Hyper-parameter Tunning
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    
    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    # model_registry
    model_registry("nyc-taxi-experiment-prefect", "2")


main_flow("2024", "01", "02")
