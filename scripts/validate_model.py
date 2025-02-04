#%%
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/youtube-comments-sentiment-analysis.mlflow")


client = mlflow.tracking.MlflowClient()
models = client.search_registered_models()

for model in models:
    print(f"Model Name: {model.name}")
    for version in client.get_latest_versions(model.name):
        print(f"Version: {version.version}, Stage: {version.current_stage}")

#%%

# load model for the model registry
def load_model_from_registry(model_name,model_version):
    model_uri=f"models:/{model_name}/{model_version}"
    model=mlflow.pyfunc.load_model(model_uri)
    return model
model1=load_model_from_registry("yt_chrome_plugin_model_updated","1")
print("model loaded sucessfully")
# %%
