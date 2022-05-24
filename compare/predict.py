import xgboost
import lightgbm
from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# from xgbtune import tune_xgb_model

train_ = pd.read_csv("../train_parody.csv", sep=",", header = None, na_values=['NULL'])
pred_  = pd.read_csv("../test_parody.csv", sep=",", header = None, na_values=['NULL'])

X, Y = train_.iloc[:,:-1], train_.iloc[:,-1]
X2, Y2 = pred_.iloc[:,:-1], pred_.iloc[:,-1]

res_ = pd.read_csv("../result_parody.csv", sep=",", header = None, na_values=['NULL'])
R = res_.iloc[:,-1]

table = '''    
| Regressor | RMSE |    
| --- | --- |
'''    

table += '| My boosted random forest |'

RMSE = mean_squared_error(R, Y2)

table += ' %.5f |\n' % RMSE


table += '| xgboost.XGBRegressor |'

model = xgboost.XGBRegressor(n_estimators = 500, max_depth = 10, n_jobs = 7)
model.fit(X, Y, eval_metric='rmse')

Y2_pred = model.predict(X2)
RMSE = mean_squared_error(Y2_pred, Y2)

table += ' %.5f |\n' % RMSE



table += '| sklearn.ensemble.GradientBoostingRegressor |'

model = GradientBoostingRegressor(n_estimators = 500, max_depth = 10, min_samples_split = 2)
model.fit(X, Y)

Y2_pred = model.predict(X2)
RMSE = mean_squared_error(Y2_pred, Y2)

table += ' %.5f |\n' % RMSE



table += '| sklearn.ensemble.RandomForestRegressor |'

model = RandomForestRegressor(n_estimators = 500, max_depth = 10, min_samples_split = 2)
model.fit(X, Y)

Y2_pred = model.predict(X2)
RMSE = mean_squared_error(Y2_pred, Y2)

table += ' %.5f |\n' % RMSE



table += '| lightgbm.LGBMRegressor |'

model = lightgbm.LGBMRegressor(n_estimators = 500, max_depth = 10, n_jobs = 7) # LGBMClassifier()
model.fit(X, Y, eval_metric='rmse')

Y2_pred = model.predict(X2)
RMSE = mean_squared_error(Y2_pred, Y2)

table += ' %.5f |\n' % RMSE



table += '| catboost.CatBoostRegressor |'

model = CatBoostRegressor(iterations = 500,
                          learning_rate = 0.02,
                          depth = 10,
                          eval_metric='RMSE')

model.fit(X, Y)

Y2_pred = model.predict(X2)
RMSE = mean_squared_error(Y2_pred, Y2)

table += ' %.5f |\n' % RMSE

with open('result_table.txt', 'w') as out:
    out.write(table)
