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
from prefect import task,flow
from mlflow.tracking import MlflowClient


@task(name = "Read Data", retries = 4, retry_delay_seconds = [1, 4, 8, 16])
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

@task(name = "Add features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  # 'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(name="Train Model")
def train_model(X_train, X_val, y_train, y_val, params, dv, run_name):
    """Train a model with given parameters and return rmse"""
    with mlflow.start_run(run_name=run_name):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_params(params)
        pathlib.Path("models").mkdir(exist_ok=True)
        with open(f"models/preprocessor_{run_name}.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(f"models/preprocessor_{run_name}.b", artifact_path="preprocessor")
        return rmse, mlflow.active_run().info.run_id
    

# Define task to register models in the Model Registry
@task(name="Registry Model")
def register_model(run_id, model_name, alias):
    """Register the model with the specified alias"""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, alias, model_version.version)
    return model_version.version

# Define flow to train two models, compare them, and assign aliases
@flow(name="Model Comparison")
def model_comparison(year: str, month_train: str, month_val: str) -> None:
    """Flow that trains two models and assigns @champion and @challenger aliases"""
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"

    # Initialize MLflow
    dagshub.init(url="https://dagshub.com/colome8/nyc-taxi-time-prediction.mlflow", mlflow=True)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    # Load data
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Add features
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Hyperparameters for both models
    params1 = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'seed': 42
    }

    params2 = {
        'max_depth': 12,
        'learning_rate': 0.01,
        'objective': 'reg:squarederror',
        'seed': 42
    }

    # Train two models
    rmse1, run_id1 = train_model(X_train, X_val, y_train, y_val, params1, dv, run_name="model1")
    rmse2, run_id2 = train_model(X_train, X_val, y_train, y_val, params2, dv, run_name="model2")

    # Register models and assign aliases
    if rmse1 < rmse2:
        champion_version = register_model(run_id1, "nyc-taxi-prefect-model", "champion")
        challenger_version = register_model(run_id2, "nyc-taxi-prefect-model", "challenger")
    else:
        champion_version = register_model(run_id2, "nyc-taxi-prefect-model", "champion")
        challenger_version = register_model(run_id1, "nyc-taxi-prefect-model", "challenger")

    print(f"Champion version: {champion_version}, Challenger version: {challenger_version}")

# Run the new flow
model_comparison("2024", "01", "02")
