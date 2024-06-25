import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

base = 'C:\\Personal Projects\\RenewableEnergy PredictionAPP\\app'
model_path = f'{base}\\model2.pkl'
reg = xgb.XGBRegressor()
model = pickle.load(open(model_path, "rb"))
#model = reg.load_model(f'{base}/model2.json')

def create_f(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    
    return df


def create_pd(fr,to):
    new = pd.date_range(fr+' 12:00:00+00:00',to+' 12:00:00+00:00', freq='10min')
    new = pd.DataFrame(index=new)
    new = create_f(new)
    return new
