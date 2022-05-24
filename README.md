# Boosted random forest

This is the experimental algorithm based on decision trees.

## Selftest

`./brf.o --selftest 0`

## Comparison of results

| Regressor | RMSE |        
| --- | --- |    
| my boosted random forest | 0.08660 |    
| xgboost.XGBRegressor | 0.08976 |    
| sklearn.ensemble.GradientBoostingRegressor | 0.07946 |    
| sklearn.ensemble.RandomForestRegressor | 0.08311 |    
| lightgbm.LGBMRegressor | 0.07557 |    
| catboost.CatBoostRegressor | 0.06097 |

## Tree 

Example of tree visualization:

`xdot forest_parody.dot_MAIN_EPOCH_1.dot`

![dot.png](pictures/dot.png)





