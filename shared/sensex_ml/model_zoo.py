from sklearn.linear_model import Ridge,Lasso,ElasticNet,HuberRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,QuantileTransformer,PowerTransformer

def preprocessors():
    return {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler(),
        'quantile_normal': QuantileTransformer(output_distribution='normal',random_state=42),
        'quantile_uniform': QuantileTransformer(output_distribution='uniform',random_state=42),
        'power_yj': PowerTransformer(method='yeo-johnson',standardize=True),
    }

def build_models(profile='full'):
    small=profile in {'smoke','daily'}
    m={}
    m['ridge']=(Ridge(),{'model__alpha':[1,20,100] if not small else [1,20]})
    m['lasso']=(Lasso(max_iter=20000),{'model__alpha':[1e-4,1e-3,1e-2] if not small else [1e-3]})
    m['elasticnet']=(ElasticNet(max_iter=20000),{'model__alpha':[1e-4,1e-3,1e-2],'model__l1_ratio':[.1,.5,.9]} if not small else {'model__alpha':[1e-3],'model__l1_ratio':[.5]})
    m['huber']=(HuberRegressor(max_iter=1000),{'model__epsilon':[1.1,1.35,1.75],'model__alpha':[1e-5,1e-3,.1]} if not small else {'model__epsilon':[1.35],'model__alpha':[1e-3]})
    m['svr']=(SVR(),{'model__C':[.1,1,10,50],'model__epsilon':[.001,.01,.05],'model__gamma':['scale','auto']} if not small else {'model__C':[1,10],'model__epsilon':[.01],'model__gamma':['scale']})
    m['knn']=(KNeighborsRegressor(),{'model__n_neighbors':[5,10,20,40],'model__weights':['uniform','distance'],'model__p':[1,2]} if not small else {'model__n_neighbors':[10,20],'model__weights':['distance'],'model__p':[2]})
    rf_grid={'model__n_estimators':[200,400],'model__max_depth':[None,8,16],'model__min_samples_leaf':[1,5,15],'model__max_features':['sqrt',.7]}
    rf_small={'model__n_estimators':[200],'model__max_depth':[None,12],'model__min_samples_leaf':[1,5],'model__max_features':['sqrt']}
    m['random_forest']=(RandomForestRegressor(random_state=42,n_jobs=-1),rf_grid if not small else rf_small)
    m['extra_trees']=(ExtraTreesRegressor(random_state=42,n_jobs=-1),rf_grid if not small else rf_small)
    gb_grid={'model__n_estimators':[200,500],'model__learning_rate':[.02,.05,.1],'model__max_depth':[2,3],'model__min_samples_leaf':[5,20]}
    gb_small={'model__n_estimators':[200],'model__learning_rate':[.05,.1],'model__max_depth':[2,3],'model__min_samples_leaf':[5]}
    m['gradient_boosting']=(GradientBoostingRegressor(random_state=42),gb_grid if not small else gb_small)
    hgb_grid={'model__max_iter':[200,500],'model__learning_rate':[.02,.05,.1],'model__max_leaf_nodes':[15,31],'model__l2_regularization':[0,1,10],'model__min_samples_leaf':[10,30,60]}
    hgb_small={'model__max_iter':[200],'model__learning_rate':[.05,.1],'model__max_leaf_nodes':[15,31],'model__l2_regularization':[0,1],'model__min_samples_leaf':[20,60]}
    m['hist_gradient_boosting']=(HistGradientBoostingRegressor(random_state=42),hgb_grid if not small else hgb_small)
    try:
        from xgboost import XGBRegressor
        xgb_grid={'model__n_estimators':[300,600],'model__max_depth':[2,3,5],'model__learning_rate':[.02,.05,.1],'model__subsample':[.8,1.0],'model__colsample_bytree':[.7,1.0],'model__min_child_weight':[1,5,15],'model__reg_lambda':[1,10],'model__reg_alpha':[0,.1]}
        xgb_small={'model__n_estimators':[200,400],'model__max_depth':[2,3],'model__learning_rate':[.03,.07],'model__subsample':[.8,1.0],'model__colsample_bytree':[.8,1.0],'model__min_child_weight':[1,5],'model__reg_lambda':[1,10]}
        m['xgboost']=(XGBRegressor(objective='reg:squarederror',random_state=42,n_jobs=-1,tree_method='hist',verbosity=0),xgb_grid if not small else xgb_small)
    except ImportError:
        pass
    return m
