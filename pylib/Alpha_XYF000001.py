#codeing:utf-8
import sys
sys.path.append("/datas/share/Msimrelease_stkbar/lib/")
from pybase_model import hfs_base_model
from pyxmlparser import *
import numpy as np
from hfs_data_api_py import * 
from pydata_api import *
from pydata import *
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO,
					format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PYMODEL(hfs_base_model):
	def __init__(self, cfg):
		super(PYMODEL ,self).__init__(cfg)
		logger.info('call PYMODEL.__init__')
		logger.info(f'tkrsize:{len(self.tkrs)}')
		cfg_root = cfg.getRoot()

		'''
		load data from binary
		'''
		logger.info(f"DATA:{cfg_root.getChild('Macros').getAttrDefault('DATA','')}")
		DATA = cfg_root.getChild('Macros').getAttrDefault('DATA','')
		self.pydpi = hfs_data_api_py(DATA)
		self.tkridx = self.pydpi.get_cidx()
		modNode = cfg.getChild("Config");
		startdate = modNode.getAttrDefault("startdate", 20040101)
		enddate	= modNode.getAttrDefault("enddate",	20210730)
		
		
		###### 基本面
		### 基本面数据库
		stkData=self.pydpi.load_pkl_data_file('Wind.AShareCashFlow.pkl')
		stkData=stkData[['STATEMENT_TYPE','S_INFO_WINDCODE','REPORT_PERIOD','ANN_DT','RECP_TAX_RENDS']] # 'CASH_RECP_SG_AND_RS'
		stkData=stkData.dropna(axis=0,subset=['ANN_DT'])
		stkData['ANN_DT']=stkData['ANN_DT'].astype('int')
		stkData=stkData[stkData['ANN_DT']>20040101]
		stkData=stkData[~stkData['S_INFO_WINDCODE'].str.contains('BJ')]
		stkData=stkData[~stkData['S_INFO_WINDCODE'].str.contains('A')]
		stkData=stkData[stkData['STATEMENT_TYPE']==408001000]
		
		### 提取变量
		stkData=stkData.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'])
		# stkData=stkData.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='last')
		stkData=stkData.reset_index()	
		stkData['variable'] = stkData['RECP_TAX_RENDS']
		print(stkData)
		
		### 计算因子
		# stkData['lag1_variable'] = stkData.groupby('S_INFO_WINDCODE').shift(1)['variable']
		# stkData['lag4_variable'] = stkData.groupby('S_INFO_WINDCODE').shift(4)['variable']
		# stkData['growth']=stkData['variable']-stkData['lag1_variable']
		# stkData['variable'] = stkData['growth']
		stkData['std']= stkData.groupby('S_INFO_WINDCODE')['variable'].rolling(4, min_periods=4).std().values
		stkData['avg']= stkData.groupby('S_INFO_WINDCODE')['variable'].rolling(4, min_periods=4).mean().values
		stkData['factor_fdm']= (stkData['variable'] - stkData['avg']) / stkData['std']
		print(stkData)
		
		### winsor
		stkData['factor_fdm']=stkData['factor_fdm'].replace([np.inf,-np.inf],np.NaN)
		# stkData['factor_fdm']=np.where((stkData['factor_fdm']>0),np.log(1+stkData['factor_fdm']),stkData['factor_fdm'])
		# stkData['factor_fdm']=np.where((stkData['factor_fdm']<0),(-1)*np.log(abs(stkData['factor_fdm']-1)),stkData['factor_fdm'])		
		stkData['factor_fdm']=np.where((stkData['factor_fdm']>10),10,stkData['factor_fdm'])
		stkData['factor_fdm']=np.where((stkData['factor_fdm']< (-10)),(-10),stkData['factor_fdm'])
		stkData['TRD_DATE'] = stkData['ANN_DT']
		stkData=stkData[['S_INFO_WINDCODE','TRD_DATE','factor_fdm']]
		print("stkData",len(stkData['factor_fdm']) - stkData['factor_fdm'].isnull().sum())
		print("stkData MEAN: ",stkData['factor_fdm'].mean())
		print("stkData STD: ",stkData['factor_fdm'].std())
		print("stkData 5 percent: ",stkData['factor_fdm'].quantile(.05))
		print("stkData 95 percent: ",stkData['factor_fdm'].quantile(.95))
		print("stkData MIN: ",stkData['factor_fdm'].min())
		print("stkData MAX: ",stkData['factor_fdm'].max())
		# print(stkData)
		stkData['factor'] = stkData['factor_fdm']
		
		# stkData['date_lag']=stkData.groupby(['S_INFO_WINDCODE'])['ANN_DT'].shift(1)
		# stkData['date_for']=stkData.groupby(['S_INFO_WINDCODE'])['ANN_DT'].shift(-1)
		# stkData['factor_for']=stkData.groupby(['S_INFO_WINDCODE'])['S_FA_ROIC_YEARLY'].shift(-1)
		# print(stkData[stkData['ANN_DT']==stkData['date_for']]['STATEMENT_TYPE']) # ['S_FA_ROIC_YEARLY'] - stkData[stkData['ANN_DT']==stkData['date_for']]['factor_for']
		# print(stkData[stkData['ANN_DT']==stkData['date_lag']])
		
		# ###### 基本面
		# ### 基本面数据库
		# stkData=self.pydpi.load_pkl_data_file('Wind.AShareFinancialIndicator.pkl')
		# ### 基本面变量
		# stkData=stkData[['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD','S_FA_ROIC_YEARLY']] # ,'STATEMENT_TYPE'
		# # fixed filter
		# stkData=stkData.dropna(axis=0,subset=['ANN_DT'])
		# stkData['ANN_DT']=stkData['ANN_DT'].astype('int')
		# stkData=stkData[stkData['ANN_DT']>startdate]
		# stkData=stkData[stkData['ANN_DT']<enddate]
		# # stkData=stkData[stkData['STATEMENT_TYPE']==408001000]
		# stkData=stkData[~stkData['S_INFO_WINDCODE'].str.contains('BJ')]
		# stkData=stkData[~stkData['S_INFO_WINDCODE'].str.contains('A')]
		# stkData['date']=pd.to_datetime(stkData['ANN_DT'].astype(str)) # REPORT_PERIOD
		# stkData['year']=stkData['date'].dt.year
		# stkData['month']=stkData['date'].dt.month
		# stkData['day']=stkData['date'].dt.day
		# stkData['TRD_DATE']=stkData['year']*10000+stkData['month']*100+stkData['day']
		# stkData['TRD_DATE']=stkData['TRD_DATE'].astype(int)
		# stkData=stkData.sort_values(by=['S_INFO_WINDCODE','TRD_DATE'])
		# stkData=stkData.reset_index()	
		# stkData=stkData.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='last')
		# stkData=stkData.reset_index()	
		# stkData['lag4_S_FA_ROIC_YEARLY'] = stkData.groupby('S_INFO_WINDCODE')['S_FA_ROIC_YEARLY'].shift(4)
		# # stkData['growth_EBITDA']=(stkData['EBITDA']-stkData['lag4_EBITDA']) / (abs(stkData['lag4_EBITDA']))	
		# # stkData['factor_fdm']= stkData.set_index('ANN_DT').groupby('S_INFO_WINDCODE')['growth_EBITDA'].rolling(4, min_periods=4).std().values
		# # stkData['rank_fdm']= stkData.groupby('ANN_DT')['growth_EBITDA'].rank() 
		# # stkData['count_fdm']= stkData.groupby('ANN_DT')['rank_fdm'].transform(max)
		# # stkData['factor_fdm'] = stkData['rank_fdm']/stkData['count_fdm'] - 0.5
		# stkData['factor_fdm'] = (stkData['S_FA_ROIC_YEARLY']-stkData['lag4_S_FA_ROIC_YEARLY'])
		
		# stkData['factor_fdm']=stkData['factor_fdm'].replace([np.inf,-np.inf],np.NaN)
		# stkData['factor_fdm']=np.where((stkData['factor_fdm']>0),np.log(1+stkData['factor_fdm']),stkData['factor_fdm'])
		# stkData['factor_fdm']=np.where((stkData['factor_fdm']<0),(-1)*np.log(abs(stkData['factor_fdm']-1)),stkData['factor_fdm'])		
		# stkData['factor_fdm']=np.where((stkData['factor_fdm']>6),6,stkData['factor_fdm'])
		# stkData['factor_fdm']=np.where((stkData['factor_fdm']< (-6)),(-6),stkData['factor_fdm'])
		# stkData['factor'] = stkData['factor_fdm']
		
		# print("stkData",len(stkData['factor_fdm']) - stkData['factor_fdm'].isnull().sum())
		# print("stkData",len(stkData['rank_fdm']) - stkData['rank_fdm'].isnull().sum())
		# print("stkData",len(stkData['count_fdm']) - stkData['count_fdm'].isnull().sum())
		# print(stkData['rank_fdm'],stkData['count_fdm'],stkData['factor_fdm'])
		
		# ##### 量价
		# ## 量价变量
		# clse = self.pydpi.load_daily_data_file('clse_adj')
		# clse = self.memmap_2_df(clse)
		# clse_data = clse.unstack()
		# clse_data=pd.DataFrame(clse_data)
		# clse_data=clse_data.reset_index()
		# clse_data.columns=['S_INFO_WINDCODE','date','S_DQ_ADJCLOSE']
		# clse_data=pd.DataFrame(clse_data)
		# clse_data['year']=clse_data['date'].dt.year
		# clse_data['month']=clse_data['date'].dt.month
		# clse_data['day']=clse_data['date'].dt.day
		# clse_data['TRD_DATE']=clse_data['year']*10000+clse_data['month']*100+clse_data['day']
		# clse_data['TRD_DATE']=clse_data['TRD_DATE'].astype(int)
		# clse_data=clse_data[clse_data['TRD_DATE']>20040101]
		# clse_data=clse_data[~clse_data['S_INFO_WINDCODE'].str.contains('BJ')]
		# clse_data=clse_data[~clse_data['S_INFO_WINDCODE'].str.contains('A')]
		
		# clse_data=clse_data[['S_INFO_WINDCODE','TRD_DATE','S_DQ_ADJCLOSE']]
		# clse_data=clse_data.sort_values(by=['S_INFO_WINDCODE','TRD_DATE'])
		# clse_data=clse_data.reset_index()
		
		# ## 量价计算
		# # clse_data['S_DQ_ADJPRECLOSE'] = clse_data.groupby('S_INFO_WINDCODE').shift(1)['S_DQ_ADJCLOSE']
		# # clse_data['ret']=clse_data['S_DQ_ADJCLOSE']/clse_data['S_DQ_ADJPRECLOSE']-1
		# # clse_data=clse_data.sort_values(by=['S_INFO_WINDCODE','TRD_DATE'])
		# # clse_data=clse_data.reset_index()
		# # clse_data['std_ret']= clse_data.set_index('TRD_DATE').groupby('S_INFO_WINDCODE')['ret'].rolling(20, min_periods=20).std().values
		# # clse_data['rank_quant']= clse_data.groupby('TRD_DATE')['std_ret'].rank() 
		# # clse_data['count_quant']= clse_data.groupby('TRD_DATE')['rank_quant'].transform(max)
		# # clse_data['factor_quant'] = clse_data['rank_quant']/clse_data['count_quant'] - 0.5
		
		
		# clse_data['std_prc']= clse_data.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].rolling(100, min_periods=20).std().values
		# clse_data['avg_prc']= clse_data.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].rolling(100, min_periods=20).mean().values
		# clse_data['factor_quant'] = (clse_data['S_DQ_ADJCLOSE']-clse_data['avg_prc'])/clse_data['std_prc']
		# print("clse_data",len(clse_data['factor_quant']) - clse_data['factor_quant'].isnull().sum())
		# print("clse_data MEAN: ",clse_data['factor_quant'].mean())
		# print("clse_data STD: ",clse_data['factor_quant'].std())
		# print("clse_data 5 percent: ",clse_data['factor_quant'].quantile(.05))
		# print("clse_data 95 percent: ",clse_data['factor_quant'].quantile(.95))
		# print("clse_data MIN: ",clse_data['factor_quant'].min())
		# print("clse_data MAX: ",clse_data['factor_quant'].max())
		# # print(clse_data)
		# clse_data=clse_data[['S_INFO_WINDCODE','TRD_DATE','factor_quant']]
		
		# ##### factor计算
		# ## 基本面触发交易
		# clse_data=stkData.merge(clse_data,left_on=('S_INFO_WINDCODE','TRD_DATE'),right_on=('S_INFO_WINDCODE','TRD_DATE'),how='left')
		# clse_data['factor_quant'] = clse_data.set_index('TRD_DATE').groupby('S_INFO_WINDCODE')['factor_quant'].ffill().values
		# ## 量价触发交易
		# # clse_data=clse_data.merge(stkData,left_on=('S_INFO_WINDCODE','TRD_DATE'),right_on=('S_INFO_WINDCODE','TRD_DATE'),how='left')
		# # clse_data['factor_fdm'] = clse_data.set_index('TRD_DATE').groupby('S_INFO_WINDCODE')['factor_fdm'].ffill().values
		
		# clse_data['factor'] = clse_data['factor_fdm']  # - clse_data['factor_quant']
		# print("merge",len(clse_data['factor_fdm']) - clse_data['factor_fdm'].isnull().sum())
		# print("merge",len(clse_data['factor_quant']) - clse_data['factor_quant'].isnull().sum())
		# print("merge",len(clse_data['factor']) - clse_data['factor'].isnull().sum())
		# print("merge",len(clse_data['TRD_DATE']) - clse_data['TRD_DATE'].isnull().sum())
		# print("merge",len(clse_data['S_INFO_WINDCODE']) - clse_data['S_INFO_WINDCODE'].isnull().sum())
		
		# ## winsor or truncate
		# clse_data['factor']=clse_data['factor'].replace([np.inf,-np.inf],np.NaN)
		# clse_data['factor']=np.where((clse_data['factor']>  10), 10,clse_data['factor'])
		# clse_data['factor']=np.where((clse_data['factor']< -10),-10,clse_data['factor'])
		# print("factor MEAN: ",clse_data['factor'].mean())
		# print("factor STD: ",clse_data['factor'].std())
		# print("factor 5 percent: ",clse_data['factor'].quantile(.05))
		# print("factor 95 percent: ",clse_data['factor'].quantile(.95))
		# print("factor MIN: ",clse_data['factor'].min())
		# print("factor MAX: ",clse_data['factor'].max())
		# print(clse_data)
		
		##### 赋值
		self.clse_data=stkData
		self.clse_data=self.clse_data.drop_duplicates(['S_INFO_WINDCODE','TRD_DATE'],keep='last')
		# print("final",len(clse_data['S_INFO_WINDCODE']) - clse_data['S_INFO_WINDCODE'].isnull().sum())
		self.alpha_data = self.clse_data.pivot(index='TRD_DATE',columns='S_INFO_WINDCODE',values='factor')
		# print(self.alpha_data)
		self.alpha_data = self.alpha_data.reindex(columns = self.tkrs)
		# print(self.alpha_data)
		
		# self.stkData=stkData
		# self.stkData=self.stkData.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='last')
		
		# self.alpha_data = self.stkData.pivot(index='ANN_DT',columns='S_INFO_WINDCODE',values='factor')
		# #print(self.alpha_data)
		# self.alpha_data = self.alpha_data.reindex(columns = self.tkrs)
		# #print(self.alpha_data)
		
		
		date_ls=[int(date_time[:-6]) for date_time in self.pydpi.daily_time_series]
		self.alpha_data=self.alpha_data.reindex(date_ls)
		self.alpha_data=self.alpha_data.ffill()
		#print(self.alpha_data)
		
		'''
		load necessary config from xml
		'''
		ndayX = modNode.getAttrDefault("nday", 10);
		logger.info(f"running with config==>ndayX:{ndayX}")
		logger.info("done with initialization")

	def memmap_2_df(self, data, min=False):
		stkNum = len(self.tkrs)
		dateNum = len(self.pydpi.daily_time_series)
		minTimeNum = len(self.pydpi.time_series)
		if min:
			reach_df = pd.DataFrame(data.reshape((dateNum * minTimeNum, stkNum)))
			time_ls = []
			for date in self.pydpi.daily_time_series:
				for time in self.pydpi.time_series:
					time_ls.append(date[:-6] + time)
			reach_df.index = pd.to_datetime(time_ls)
			reach_df.columns = self.tkrs
		else:
			reach_df = pd.DataFrame(data)
			reach_df.index = self.pydpi.daily_time_series
			reach_df.index = pd.to_datetime([x[:-6] for x in self.pydpi.daily_time_series])
			reach_df.columns = self.tkrs
		return reach_df
		
	def push_interval_data(self, didx, tidx):
		#print(didx,tidx)
		date = self.pydpi.daily_time_series[didx][:-6]
		#print(date)
		if int(date) in self.alpha_data.index:
			wts =self.alpha_data.loc[int(date)] ##筛选出ANN_DT是这一天的
		else:
			wts = [np.nan]*len(self.tkrs) ##没有这一天的数据
		t = self.pydpi.time_series[tidx]

		wts =wts
		self.wts = wts # wts is float pointer, do not perform any operator directly on wts
		debug = False
		if debug:
			logger.info("alpha stats: Min({:.2f}), Q1({:.2f}), Mean({:.2f}), Median({:.2f}), Q9({:.2f}),Max({:.2f}), Cnt({:.0f})".format(
				np.nanmin(wts), np.nanquantile(wts, 0.1), np.nanmean(wts), np.nanmedian(wts),
				np.nanquantile(wts, 0.9),np.nanmax(wts),len(self.tkrs) - np.isnan(wts).sum()
			))
		return True

	def __del__(self):
		print("call __del__")


def create(cfg):
	return PYMODEL(cfg)

