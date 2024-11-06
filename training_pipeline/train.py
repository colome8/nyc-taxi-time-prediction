import os
import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor


@task(name="Read Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


@task(name="Add Feature")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(name="Hyper-parameter Tuning")
def hyper_parameter_tuning(X_train, X_val, y_train, y_val, dv):
    # Desactivar el autologging para evitar la advertencia
    # Puedes reactivar funciones específicas si lo deseas
    mlflow.sklearn.autolog(log_input_examples=False)

    def objective(params):
        with mlflow.start_run(nested=True):
            # Tag model
            mlflow.set_tag("model_family", "random_forest")

            # Train model
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)

            # Predict in the validation dataset
            y_pred = rf.predict(X_val)

            # Calculate metric
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            # Log performance metric
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    with mlflow.start_run(run_name="Random-Forest Hyper-parameter Optimization", nested=True):
        search_space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),
            'random_state': 42
        }

        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )

        # Convert params back to integers where necessary
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["min_samples_split"] = int(best_params["min_samples_split"])
        best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
        best_params["random_state"] = 42

        mlflow.log_params(best_params)

    return best_params


@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """Train a model with the best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best model ever"):
        rf = RandomForestRegressor(**best_params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(rf, artifact_path="models")

    return None


@task(name="Register Best Model")
def register_model(tracking_uri: str, experiment_name: str, model_name: str, alias: str) -> None:
    """Register the best model in the MLflow Model Registry and assign an alias."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)

    # Obtener el experimento por nombre
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Buscar los runs ordenados por RMSE ascendente
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in the experiment.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id

    # Definir el URI del run para registrar el modelo
    # Asegúrate de que el artifact_path coincide con cómo loggeaste el modelo
    run_uri = f"runs:/{best_run_id}/models/model"

    # Registrar el modelo en el Model Registry
    result = client.register_model(
        model_uri=run_uri,
        name=model_name
    )

    # Asignar el alias '@champion' a la versión registrada
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=result.version
    )

    # Opcional: Actualizar la descripción de la versión del modelo
    client.update_model_version(
        name=model_name,
        version=result.version,
        description=f"Model version {result.version} registered as '{alias}' with RMSE: {best_run.data.metrics['rmse']:.4f}"
    )

    print(f"Model '{model_name}' version {result.version} registered and aliased as '{alias}'.")


@flow(name="Main Flow")
def main_flow(year: int, month_train: int, month_val: int) -> None:
    """The main training pipeline"""

    # Obtener la ruta absoluta del directorio donde se encuentra este script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir las rutas absolutas de los archivos de datos
    train_path = os.path.join(script_dir, '..', 'data', f'green_tripdata_{year}-{month_train:02d}.parquet')
    val_path = os.path.join(script_dir, '..', 'data', f'green_tripdata_{year}-{month_val:02d}.parquet')

    # MLflow settings
    dagshub.init(repo_owner='JuanPab2009', repo_name='nyc-taxi-time-prediction', mlflow=True)

    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Hyper-parameter Tuning
    best_params = hyper_parameter_tuning(X_train, X_val, y_train, y_val, dv)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    # Registrar el mejor modelo en el Model Registry
    register_model(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name="nyc-taxi-experiment-prefect",
        model_name="nyc-taxi-model-prefect",
        alias="champion"
    )


if __name__ == "__main__":
    main_flow(year=2024, month_train=1, month_val=2)