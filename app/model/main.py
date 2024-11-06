import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient


# MLflow settings
dagshub_repo = "https://dagshub.com/JuanPab2009/nyc-taxi-time-prediction"

# MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
MLFLOW_TRACKING_URI = "https://dagshub.com/JuanPab2009/nyc-taxi-time-prediction.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

run_ = mlflow.search_runs(order_by=['metrics.rmse ASC'],
                          output_format="list",
                          experiment_names=["nyc-taxi-experiment"]
                          )[0]

run_id = run_.info.run_id

run_uri = f"runs:/{run_id}/preprocessor"

client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

model_name = "nyc-taxi-model"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)
def preprocess(input_data):

    input_dict = {
        'PU_DO': input_data.PULocationID + "_" + input_data.DOLocationID,
        'trip_distance': input_data.trip_distance,
    }

    return dv.transform(input_dict)

def predict(input_data):

    X_pred = preprocess(input_data)

    return champion_model.predict(X_pred)

app = FastAPI()

class InputData(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]
    return {"prediction": float(result)}