import mlflow
logged_model = 'runs:/6568bb3363534272b22a9f628018d43a/lgbm_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))