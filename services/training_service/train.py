import argparse,json,time,warnings
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sensex_ml.config import RAW_PATH,MODEL_DIR,REPORT_DIR,RANDOM_STATE,N_SPLITS,TEST_FRACTION,N_JOBS
from sensex_ml.data import load_ohlcv
from sensex_ml.features import make_features,feature_columns
from sensex_ml.model_zoo import build_models,preprocessors
warnings.filterwarnings('ignore')

def metrics(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'DirectionalAccuracy':float(np.mean(np.sign(y)==np.sign(p)))}
def split(X,y,f):
 n=max(100,int(len(X)*f)); return X.iloc[:-n],X.iloc[-n:],y.iloc[:-n],y.iloc[-n:]
def grids(profile,model_params):
 ps=preprocessors(); out=[]
 scalers=['standard','robust','minmax','quantile_normal','quantile_uniform','power_yj'] if profile=='full' else ['standard','robust','minmax','power_yj']
 if profile=='smoke': scalers=['standard']
 for name in scalers+['none']:
  g={'imputer__strategy':['median','mean'] if profile=='full' else ['median'],'variance__threshold':[0.0,1e-10] if profile=='full' else [0.0],'scaler':[ps[name]] if name!='none' else [None]}
  g.update(model_params); out.append(g)
 return out

def train_model(name,model,params,X,y,Xte,yte,tscv,profile):
 pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('variance',VarianceThreshold(0.0)),('scaler',None),('model',model)])
 gs=GridSearchCV(pipe,grids(profile,params),scoring='neg_mean_absolute_error',cv=tscv,n_jobs=N_JOBS,refit=True,return_train_score=False,error_score='raise',verbose=0)
 t=time.time(); gs.fit(X,y); p=gs.predict(Xte); return gs,{'model':name,'best_params':gs.best_params_,'cv_best_MAE':float(-gs.best_score_),'cv_fits':len(gs.cv_results_['params']),'search_seconds':time.time()-t,**metrics(yte,p)}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','daily','full'],default='full'); ap.add_argument('--splits',type=int,default=N_SPLITS); ap.add_argument('--test-fraction',type=float,default=TEST_FRACTION); ap.add_argument('--max-models',type=int); a=ap.parse_args()
 raw=load_ohlcv(RAW_PATH); feat=make_features(raw); cols=feature_columns(feat)
 d=feat[cols+['Target_Open_Return','Target_Close_Return']].replace([np.inf,-np.inf],np.nan).dropna(subset=['Target_Open_Return','Target_Close_Return'])
 X=d[cols]; yo=d.Target_Open_Return; yc=d.Target_Close_Return
 Xtr,Xte,yo_tr,yo_te=split(X,yo,a.test_fraction); _,_,yc_tr,yc_te=split(X,yc,a.test_fraction); cv=TimeSeriesSplit(n_splits=a.splits,gap=1)
 zoo=build_models(a.mode); 
 if a.max_models: zoo=dict(list(zoo.items())[:a.max_models])
 rows=[]; fitted={}
 for target,ytr,yte in [('open_return',yo_tr,yo_te),('close_return',yc_tr,yc_te)]:
  base=DummyRegressor(strategy='mean').fit(Xtr,ytr); rows.append({'target':target,'model':'dummy_mean','best_params':{'strategy':'mean'},'cv_best_MAE':None,'cv_fits':1,'search_seconds':0,**metrics(yte,base.predict(Xte))}); fitted[target+'__dummy_mean']=base
 for name,(model,params) in zoo.items():
  for target,ytr,yte in [('open_return',yo_tr,yo_te),('close_return',yc_tr,yc_te)]:
   print(f'[{a.mode}] {name} / {target}',flush=True); gs,row=train_model(name,model,params,Xtr,ytr,Xte,yte,cv,a.mode); row['target']=target; rows.append(row); fitted[target+'__'+name]=gs.best_estimator_
 lb=pd.DataFrame(rows); lb['rank_by_MAE']=lb.groupby('target').MAE.rank(method='min'); lb=lb.sort_values(['target','MAE','RMSE']).reset_index(drop=True); lb.to_csv(REPORT_DIR/'leaderboard.csv',index=False)
 best={}
 for target in ['open_return','close_return']:
  s=lb[lb.target==target].iloc[0]; name=s.model; est=fitted[target+'__'+name]
  if name!='dummy_mean':
   est.fit(X,yo if target=='open_return' else yc); path=MODEL_DIR/f'best_{target}.joblib'; joblib.dump(est,path); rel=str(path.relative_to(MODEL_DIR.parent.parent))
  else: path=None; rel=None
  best[target]={'name':name,'path':rel,'test_metrics':s.to_dict()}
 joblib.dump(cols,MODEL_DIR/'feature_columns.joblib')
 latest=feat.index[-1]; latest_feat=feat.loc[[latest],cols].replace([np.inf,-np.inf],np.nan)
 joblib.dump({'date':str(latest.date()),'close':float(raw.loc[latest,'Close']),'features':latest_feat,'feature_columns':cols},MODEL_DIR/'latest_state.joblib')
 hist=[c for c in ['Open','High','Low','Close','RSI_14','Return_std_20','ATR_pct'] if c in feat]; feat[hist].dropna().tail(1500).to_csv(REPORT_DIR/'history.csv')
 manifest={'created_at':datetime.now(timezone.utc).isoformat(),'mode':a.mode,'cv':{'type':'TimeSeriesSplit','n_splits':a.splits,'gap':1},'holdout_fraction':a.test_fraction,'rows_total':len(X),'rows_train':len(Xtr),'rows_test':len(Xte),'feature_count':len(cols),'features':cols,'models':list(zoo),'best_models':best}
 (REPORT_DIR/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
 print('\nTOP RESULTS'); print(lb[['target','model','MAE','RMSE','R2','DirectionalAccuracy','cv_fits']].groupby('target').head(6).to_string(index=False))

if __name__=='__main__': main()
