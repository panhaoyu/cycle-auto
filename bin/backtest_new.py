import pandas as pd
import numpy as np
import xlsxwriter
import os, sys
import matplotlib.pyplot as plt
import bottleneck
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import dask.dataframe as dd
from datetime import datetime
import gc
import os,sys,pdb,glob,math
import statsmodels.api as sm

dir1 = '/datas/cuijn/style_reverse.parquet'
if sys.argv[1] == '1':
	dir1 = dir1
else:
	dir1 = sys.argv[1]
if sys.argv[2] == '1':
	dir2 = '/home/cuijn'
else:
	dir2 = sys.argv[2]


#dumpAlpha = pd.read_parquet(dir1)
dumpAlpha = pd.read_csv(dir1)
dumpAlpha.loc[dumpAlpha['minTime'].astype(int) == 93900, 'minTime'] = '093900'
#print(dumpAlpha)
var = dumpAlpha.columns.tolist()[3:]
dumpAlpha['feature'] = (dumpAlpha["dateStr"].astype(str) + dumpAlpha["minTime"].astype(str)).astype(int)
temp = ['feature', 'code', 'dateStr', 'minTime']
temp.extend(var)
df_factor = dumpAlpha[temp]
dumpAlpha = dumpAlpha.set_index(['feature', 'code'])[var[0]].unstack().T
dumpAlpha = dumpAlpha.sort_index()

tkrs = list(filter(lambda x:len(x)>0, map(lambda x:x.strip().split()[0]+'.'+x.strip().split()[1], open(os.path.join("/datas2/share/alpha_bar/DestData/Meta", 'universe.txt'), 'r').readlines())))

infile = open('/datas2/share/alpha_bar/DestData/Meta/daily_time_list.txt')
timelist=[]
time_dict={}
while True:
    infile_min = open('/datas2/share/alpha_bar/DestData/Meta/time_list.txt')
    line = infile.readline()
    if not line:
        break
    while True:
        line_min = infile_min.readline()
        if not line_min:
            break
        timelist.append(line.strip('\n')[:8]+line_min.strip('\n'))

for i in range(len(timelist)):
    time_dict[i] = timelist[i]

idx = 1500
endidx = 3161

def reach_data(file_name,datatype = 'float32'):
    '''从二进制文件中读取数据'''

    clse = np.memmap('/datas2/share/alpha_bar/DestData/MinBar/'+file_name, dtype=datatype, mode='r',shape=(3160*240,4507))
    reach_df = pd.DataFrame(clse[idx*240:endidx*240][:])

    reach_df.index = timelist[idx*240:endidx*240]
    reach_df.columns = tkrs
    reach_df = reach_df.T
    reach_df.columns = [int(i) for i in reach_df.columns]
    return reach_df

df1 = reach_data('ret_1d.dat')
#print(df1)
df2 = df1[list(map(int, dumpAlpha.columns.tolist()))]
df2 = df2.loc[dumpAlpha.index.tolist()]
df2 = df2.sort_index()
df2 = df2.unstack().reset_index()
df2.columns = ['feature', 'code', 'return']
df_res = pd.merge(df2, df_factor, how = 'left')

df3 = reach_data('abret_1000.dat')
df4 = df3[list(map(int, dumpAlpha.columns.tolist()))]
df4 = df4.loc[dumpAlpha.index.tolist()]
df4 = df4.sort_index()
df4 = df4.unstack().reset_index()
df4.columns = ['feature', 'code', 'abret_1000']
df_res = pd.merge(df_res, df4, how = 'left')

def quantile_calc(x, _quantiles):
    quantiles = pd.qcut(x, _quantiles ,labels=False, duplicates = 'drop') + 1
    return quantiles.sort_index()


def getrank(x):
    #rank = pd.DataFrame(x)
    #rank.columns = ['rank']
    rank = x.rank(method = 'average', na_option='keep', pct=True) - 0.5
    #rank['rank'] = min_max_scaler.fit_transform(np.array(rank[['rank']]).reshape(-1,1))
    return rank

def getzscore(x):
    zscore = pd.DataFrame(x)
    zscore.columns = ['zscore']
    mean = np.nanmean(zscore['zscore'].tolist())
    std = np.nanstd(zscore['zscore'].tolist())
    zscore['zscore'] = (zscore['zscore'] - mean)/std
    return zscore['zscore']


def get_turnover(x):
    turnover = pd.DataFrame(x).copy()
    turnover = turnover.fillna(0)
    turnover.columns = ['orgin']
    turnover['lag1'] = turnover['orgin'].shift(1)
    turnover = turnover.fillna(0)
    turnover['turnover'] = abs(turnover['orgin'] - turnover['lag1'])
    turnover.fillna(0)
    turnover_list = turnover['turnover']
    return turnover_list.sort_index()


def get_turnover_sum(x):
    turnover = x.abs().fillna(0)
    return turnover.sort_index()


def standardize(v, method=''):
    #df1 = df
    if method == 'constant':
        # standardization with constant mean and std
        constant_len = 0.2
        const_to_add = 0.01
        
        len_to_constant = int(constant_len*len(v))
        v = v.replace([-np.inf,np.inf],np.nan)
        const_mean = np.nanmean(np.array(v.tolist()[0:len_to_constant], dtype=np.float64))
        const_std = np.nanstd(np.array(v.tolist()[0:len_to_constant], dtype=np.float64))
        print(const_mean)
        print(const_std)
        v = (v-const_mean)/(const_std + const_to_add)

    else:
        # standardization with 95% and 5% percentile
        constant_len = 0.2
        const_to_add = 0.01
        
        len_to_constant = int(constant_len*len(v))
        v = v.replace([-np.inf,np.inf],np.nan)
        per_5 = np.nanpercentile(v.tolist()[0:len_to_constant], 5)
        per_95 = np.nanpercentile(v.tolist()[0:len_to_constant], 95)
        const_mean = np.nanmean(v.tolist()[0:len_to_constant])
        #print(per_5)
        #print(per_95)
        v = (v - const_mean)/(per_95-per_5 + const_to_add)
        
    return v


def adding_random(v, size = 100):
    size = size
    np.random.seed(3)
    random = np.random.rand(len(v))/size
    v = v + random
    return v


def cleaner_trimP(df, v, trimP):
    v_list = np.array(df[v].tolist())
    v_list[v_list>trimP] = np.nan
    return v_list

def cleaner_trimN(df, v, trimN):
    v_list = np.array(df[v].tolist())
    v_list[v_list<trimN] = np.nan
    return v_list

def cleaner_winsorP(df, v, winsorP):
    v_list = np.array(df[v].tolist())
    v_list[v_list>winsorP] = winsorP
    return v_list

def cleaner_winsorN(df, v, winsorN):
    v_list = np.array(df[v].tolist())
    v_list[v_list<winsorN] = np.nan
    return v_list

def cal_weight(x):
    x = x.fillna(0)
    try:
        w1 = x[x>=0]/x[x>=0].abs().sum()
        w2 = x[x<0]/x[x<0].abs().sum()
        w3 = pd.concat([w1, w2])
        #print(w3)
        return w3.sort_index()
    except:
        return x

def get_drawdown(x):
    #print(x)
    drawdown = 0
    down = 0
    st = x['dateStr'].tolist()[0]
    start = x['dateStr'].tolist()[0]
    end = x['dateStr'].tolist()[0]
    ret_list = x['average'].tolist()
    for i in range (0, len(x)):
        if ret_list[i] < 0:
            down = down + ret_list[i]
        else:
            if down < drawdown:
                drawdown = down
                end = x['dateStr'].tolist()[i]
                start = st
                st = x['dateStr'].tolist()[i]
                down = 0
            else:
                down = 0
                st = x['dateStr'].tolist()[i]
    return (drawdown, start, end)

def b_group(x_vec,y_vec,group_N = 1000,group_function = np.mean):
## remove NA
    flag = (~np.isnan(x_vec)) & (~np.isnan(y_vec))
    x1 = x_vec[flag]
    y1 = y_vec[flag]

    ## group mean/median
    ## group_flag = cut(c(1:length(x2)),breaks=group_N) ## each group has same data range;
    group_flag = pd.qcut(x1.rank(method='first'), group_N,labels = range(group_N))
    temp = pd.DataFrame({'x1':x1,'y1':y1,'group_flag':group_flag})
    x3 = temp.groupby('group_flag')['x1'].apply(group_function)
    y3 = temp.groupby('group_flag')['y1'].apply(group_function)

    result = pd.DataFrame({'x':x3,'y':y3})
    return(result)

def b_group_plot(ax,x_vec,y_vec,group_N=1000,group_function=np.mean, weights=np.nan):
## show linear relationship between x_vec and y_vec
    a1 = b_group(x_vec,y_vec,group_N,group_function)
    #需要修改
    if ~np.isnan(weights):
        g1 = sm.OLS(y_vec,sm.add_constant(x_vec)).fit()
    else:
        g1 = sm.OLS(y_vec,sm.add_constant(x_vec)).fit()

    ax.scatter(a1.x,a1.y)
    ax.axline((0,g1.params[0]),slope = g1.params[1])

    ax.legend(loc="upper left",frameon=True,title="y="+str(g1.params[0])+"+"+str(g1.params[1])+"x")
    ax.title.set_text(f"risk ic plot")


def plot_eval(data, v, ret = 'alpha_market', weight = '', ifrank = True, nan = 0, start_date = 0, end_date = 0, complex_ver = False, trimP=0, trimN=0, winsorP=0, winsorN=0):

    df = data[['dateStr', 'minTime', 'code', 'abret_1000', ret, v]].copy()

    df['abret_1000'] = np.where(abs(df['abret_1000'])>0.5, np.nan, df['abret_1000'])
    df[ret] = np.where(abs(df[ret])>0.5, np.nan, df[ret])

    #since backtest period can have strong effect on backtest result, we can specify the backtest period by adding params start_date and end_date
    if (start_date != 0 and end_date != 0):
        df = df[df['dateStr'] > start_date]
        df = df[df['dateStr'] < end_date]
        df = df.reset_index()

    fig = plt.figure(figsize=(20,18))
   
    #trimP=0, trimN=0, winsorP=0, winsorN=0
    if trimP != 0:
        df[v] = cleaner_trimP(df, v, trimP)
    if trimN != 0:
        df[v] = cleaner_trimN(df, v, trimN)
    if winsorP != 0:
        df[v] = cleaner_winsorP(df, v, winsorP)
    if winsorN != 0:
        df[v] = cleaner_winsorN(df, v, winsorN)

    df['dateStr'] = df['dateStr'].astype(str)

    if weight == 'rank':
        df['factor'] = df.groupby(['dateStr', 'minTime'])[v].apply(getrank)
        df['factor_backtest'] = df['factor']
    elif weight == 'zscore':
        df['factor'] = df.groupby(['dateStr', 'minTime'])[v].apply(getzscore)
        df['factor'] = df.groupby(['dateStr', 'minTime'])['factor'].apply(getrank)
    else:
        df['factor'] = df[v]
        if ifrank == True:
            df['factor_backtest'] = df.groupby(['dateStr', 'minTime'])['factor'].apply(getrank)
        else:
            df['factor_backtest'] = df['factor']

    df['weight'] = df.groupby(['dateStr', 'minTime'])['factor'].apply(cal_weight)


    # construct turnover columns and factor columns
    df['turnover1'] = df.groupby(['minTime' ,'code'])['weight'].apply(get_turnover)
    df['turnover2'] = df.groupby(['minTime', 'code'])['weight'].apply(get_turnover_sum)

    turnover = (df.groupby(['dateStr', 'minTime'])['turnover1'].sum()/df.groupby(['dateStr', 'minTime'])['turnover2'].sum()).fillna(0)
    turnover[~ np.isfinite(turnover)] = 1
    #turnover = df.groupby(['dateStr', 'minTime'])['turnover1'].sum()/2
    #print(df.groupby(['dateStr', 'minTime'])['turnover1'].sum())
    #print(df.groupby(['dateStr', 'minTime'])['turnover2'].sum())
    turn_by_date = turnover.groupby('dateStr').mean()
    #turn_by_date = [x for x in turn_by_date if np.isnan(x) == False]
    #turn_by_date[0] = 0
    #print(len(df))
   

    #print(turn_by_date)

    #print(np.corrcoef(df[ret].tolist(), df['factor'].tolist()))
    ic_minTime = df.groupby(['minTime'])[ret, 'factor'].corr() # calculate IC for each day
    ic_minTime = ic_minTime[ic_minTime.index.get_level_values(1) == ret]['factor'] # get corr of ret vs v
    #print(ic_minTime)
    #ic_minTime = ic_minTime.droplevel(1) # drop index 'ret' and then we can have date + minTime + ic
    ic = round(sum(ic_minTime.tolist())/len(ic_minTime.tolist()), 4)
    
    ##figure 1
    if complex_ver:
       ax1 = fig.add_subplot(421)
       tmp = df.groupby(['dateStr'])[ret, 'factor'].corr() # calculate IC for each day
       tmp = tmp[tmp.index.get_level_values(1) == ret]['factor'] # get corr of ret vs v
       tmp = tmp.droplevel(1) # drop index 'ret' and then we can have date + ic
       # tmp.plot(title='IC by dateStr', rot=-30, figsize=(10, 6))
       tmp.plot(title='IC %s , IR %s' % (round(tmp.mean(), 3), round(tmp.mean() / tmp.std(), 3)), rot=-30, label = 'ic') # calculate avg ic and ir
       ax1.grid()
       ax1.legend()
       ic = round(tmp.mean(), 3)

    
    #df['factor_backtest'] = df['factor']
    #print('get_rank_success')
    df['long'] = df[ret][df['factor_backtest'] > 0] * df['factor_backtest'][df['factor_backtest'] > 0] # rank as weight
    df['short'] = df[ret][df['factor_backtest'] < 0] * df['factor_backtest'][df['factor_backtest'] < 0] # rank as weight

    df_tmp = df[df['factor_backtest'] > 0]
    long_sum = df_tmp.groupby(['dateStr', 'minTime'])['factor_backtest'].sum() # sum of long factor
    y_mean = df.groupby(['dateStr', 'minTime'])[ret].mean() # mean of return of all stocks for each day and mintime
    aa = df.groupby(['dateStr', 'minTime'])['long'].sum() # get long return
    longret = aa / long_sum - y_mean # return minus avg of market

    df_tmp = df[df['factor_backtest'] < 0]
    short_sum = df_tmp.groupby(['dateStr', 'minTime'])['factor_backtest'].sum()
    y_mean = df.groupby(['dateStr', 'minTime'])[ret].mean()
    aa = df.groupby(['dateStr', 'minTime'])['short'].sum()
    shortret = - aa / short_sum + y_mean

    overall_ret = longret.add(shortret,fill_value = 0)/2 # all return (market basic included)
    #??? fill 0 or y_mean, since y_mean is for calculate alpha, when we did not buy any stock, it should be 0
    # when using abret as ret, we actually get alpha of alpha


    turnover_mean = np.nanmean(turnover.tolist())
    turnover_mean2 = pd.DataFrame(turnover.groupby('minTime').mean())
    turnover_mean2.columns = ['turnover']
    turnover_mean2['IC'] = ic_minTime.tolist()
    print('turnover: {}'.format(turnover_mean))
    print('IC: {}'.format(ic))
    print('turnover and IC by minTime:')
    print(turnover_mean2)

    tmp_long = longret.unstack()
    tmp_all = overall_ret.unstack()
    

    tmp_all['average']=tmp_all.mean(axis=1)
    
    ######
    if complex_ver:
        ax2 = fig.add_subplot(422)
    else:
        ax2 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(422)

    tmp_long.mean(axis=1).cumsum().plot(label = 'longret', ax = ax2)
    tmp_all.mean(axis=1).cumsum().plot(label = 'ret', ax = ax2)

    ax2.grid()
    ax2.legend()
    ax2.title.set_text(f"longret:{round(tmp_long.mean().mean()*1e4, 1)},\
    longret_ir:{round(tmp_long.mean().mean() / tmp_long.std().mean(), 3)}, \
    ret:{round(tmp_all.mean().mean()*1e4, 1)},\
    ret_ir:{round(tmp_all.mean().mean() / tmp_all.std().mean(), 3)}")
    
    longret = round(tmp_long.mean().mean()*1e4, 1)
    returns = round(tmp_all.mean().mean()*1e4, 1)
    
    
    
    if complex_ver:
        ##########
        ax3 = fig.add_subplot(423)
        tmp_long.cumsum().plot(legend = True, ax = ax3)
        ax3.legend()
        ax3.grid()
        ax3.title.set_text(f"long return plot with t")
        ########
        ax4 = fig.add_subplot(424)
        tmp_all.cumsum().plot(legend = True, ax = ax4)
        ax4.legend()
        ax4.grid()
        ax4.title.set_text(f"overall return plot with t")

        df1 = df.dropna(subset = ['factor'])

        df1.loc[:, 'rk'] = df1.groupby(['dateStr', 'minTime'])[v].apply(quantile_calc, 10)
    
        ############
        ax5 = fig.add_subplot(425)
        t = df1.groupby(['rk'])[ret].mean()
        t.plot.bar(ax = ax5)
        #print(t)
        ax5.grid()
        ax5.title.set_text(f"Quantile ret in mean bps; tvr:{round(t.mean(), 4)}")
    
        ax6 = fig.add_subplot(426)
        t = df1.groupby(['dateStr','rk'])[ret].mean().unstack(level=1).cumsum()
        t.plot(rot=-30, ax = ax6)
        ax6.grid()
        ax6.legend()
        ax6.title.set_text(f"quantile plot")
        #t
    
        ax7 = fig.add_subplot(427)
        t = df1.groupby(['rk'])[v].mean()
        t.plot.bar(ax = ax7)
        ax7.grid()
        #print(t)
        ax7.title.set_text(f"x distribution")
   
    df = df.dropna(subset = [v])
    #print(len(df))
    df = df.reset_index(drop = True)
    df = df.fillna(nan)

    if complex_ver:
        ax8 = fig.add_subplot(428)
    else:
        ax8 = fig.add_subplot(212)
        
    #print(df)
        
    b_group_plot(ax8, df['factor'] ,df[ret], group_N=1000, group_function=np.mean, weights=np.nan)
    plt.show()
    
    plt.tight_layout()
    
    fig = plt.savefig(v + "_eval.png")
    plt.pause(1)
    
    tmp_all.to_csv(v + "_pnl.csv")

    #pnl_file = pd.DataFrame(tmp_all, index = datetime.strptime(str(tmp_all.index.tolist()),'%Y%m%d').strftime('%Y-%m-%d'), columns = ['average'])
    

    pnl_file = pd.DataFrame(tmp_all, columns = tmp_all.columns.tolist()[0:-1])
    #pnl_file['minTime'] = '[93900, 103900, 130900, 140900]'

    #pnl_file['turnover'] = turn_by_date

    #print(pnl_file)
    #pnl_file.index = list(map(lambda x: datetime.strptime(str(int(float(x))),'%Y%m%d').strftime('%Y-%m-%d'), tmp_all.index.tolist()))
    pnl_file = pnl_file.stack()
    pnl_file = pnl_file.reset_index()
    pnl_file.columns= ['dateStr', 'tme','pnl1']

    #df_longshort = df.groupby(['dateStr', 'minTime'])['factor'].sum()
    longNum = df[df['factor'] > 0].groupby(['dateStr', 'minTime'])['factor'].count()
    shortNum = df[df['factor'] < 0].groupby(['dateStr', 'minTime'])['factor'].count()
    longsum = df[df['factor'] > 0].groupby(['dateStr', 'minTime'])['factor'].sum()
    shortsum = df[df['factor'] < 0].groupby(['dateStr', 'minTime'])['factor'].sum()
    #longNum.columns= ['dateStr', 'tme','longNum']
    #shortNum.columns= ['dateStr', 'tme','shortNum']
    #longsum.columns= ['dateStr', 'tme','long']
    #shortsum.columns= ['dateStr', 'tme','short']
    df_longshort = pd.DataFrame([longsum, shortsum, longNum, shortNum]).T.reset_index()
    df_longshort.columns = ['dateStr', 'tme', 'long', 'short', 'longNum', 'shortNum']
    #df_longshort = pd.concat([longsum, shortsum['short'], longNum['longNum'], shortNum['shortNum']], axis = 1)
    #print(df_longshort)
    #df_longshort['dateStr'] = list(map(lambda x: datetime.strptime(str(int(float(x))),'%Y%m%d').strftime('%Y-%m-%d'), df_longshort['dateStr'].tolist()))
    pnl_file = pd.merge(pnl_file, df_longshort, on = ['dateStr', 'tme'], how = 'left')

    pnl_file['pnl'] = pnl_file['pnl1']

    df_turnover = turnover.reset_index()
    df_turnover.columns = ['dateStr', 'tme', 'turnover']
    #df_turnover['dateStr'] = list(map(lambda x: datetime.strptime(str(int(float(x))),'%Y%m%d').strftime('%Y-%m-%d'), df_turnover['dateStr'].tolist()))
    pnl_file = pd.merge(pnl_file, df_turnover, on = ['dateStr', 'tme'], how = 'left')

    pnl_file = pnl_file[['dateStr', 'tme', 'long', 'short', 'longNum', 'shortNum', 'pnl', 'turnover']]
    pnl_file = pnl_file.set_index('dateStr')
    
    #print(pnl_file)
    
    pnl_file.to_csv(v + "_pnl1.txt")
        
    tmp_all['year'] = [int(int(float(x))/10000) for x in tmp_all.index.tolist()]
    tmp_all['dateStr'] = tmp_all.index.tolist()

    output = pd.DataFrame(index = list(set(tmp_all['year'].tolist())), columns = ['from', 'to', 'return', 'pnl_per_day', 'win_rate', 'sharpe', 'drawdown'])
    output['return'] = tmp_all.groupby('year')['average'].sum()
    output['pnl_per_day'] = tmp_all.groupby('year')['average'].mean()
    output['from'] = tmp_all.groupby('year')['dateStr'].min()
    output['to'] = tmp_all.groupby('year')['dateStr'].max()
    output['win_rate'] = tmp_all[tmp_all['average'] > 0].groupby('year')['average'].count()/tmp_all.groupby('year')['average'].count()
    output['drawdown'] = tmp_all.groupby(['year'])['average', 'dateStr'].apply(get_drawdown)
    output[['drawdown', 'dd_start', 'dd_end']] = output['drawdown'].apply(pd.Series)
    #output['dd_start'] = tmp_all.groupby('year')['average', 'dateStr'].apply(get_drawdown)
    #output['dd_end'] = tmp_all.groupby('year')['average', 'dateStr'].apply(get_drawdown)
    output['sharpe'] = tmp_all.groupby('year')['average'].mean()/tmp_all.groupby('year')['average'].std()*np.sqrt(242)
    output['win/loss'] = -tmp_all[tmp_all['average'] > 0].groupby('year')['average'].mean()/tmp_all[tmp_all['average'] <0].groupby('year')['average'].mean()
    
    sum_ret = tmp_all['average'].mean()*242
    sum_pnl_per_day = tmp_all['average'].mean()
    sum_winrate = tmp_all[tmp_all['average'] > 0]['average'].count()/tmp_all['average'].count()
    sum_drawdown = get_drawdown(tmp_all)
    sumdd_start = sum_drawdown[1]
    sumdd_end = sum_drawdown[2]
    sum_dd = sum_drawdown[0]
    #print(sum_drawdown)
    sum_sharpe = tmp_all['average'].mean()/tmp_all['average'].std()*np.sqrt(242)
    sum_win_ret = -tmp_all[tmp_all['average'] > 0]['average'].mean()/tmp_all[tmp_all['average'] <0]['average'].mean()

    #summary = {'from': min(tmp_all['dateStr'].tolist()), 'to' : max(tmp_all['dateStr'].tolist()), 'return': 0, "pnl_per_day": 0, 'winrate': 0, 'sharpe': 0, 'drawdown': 0, 'dd_start': 0, 'dd_end': 0, 'win/loss': 0}
    output.loc['summary'] = [min(tmp_all['dateStr'].tolist()), max(tmp_all['dateStr'].tolist()), sum_ret, sum_pnl_per_day, sum_winrate, sum_sharpe, sum_dd,sumdd_start,sumdd_end,sum_win_ret]
    
    print(output)

    
    return ic, longret, returns, tmp_all


gc.collect()

path =  dir2
filename = 'test_report.xlsx'

os.chdir(path)
writer = pd.ExcelWriter(filename, engine='xlsxwriter')

diff_param = False # 提供在不同的中使用不同参数的计算方式，将此项置为True，并在param_list按顺序明确参数即可，如不需要，则置为False
flag = 0
flagn = len(var)
err = []
param= [[True, '', False, 0, 'zscore', False, 0, 0, 0, 0, 0, 0, -0.3],
       [True, '', False, 0, 'zscore', False, 0, 0, 0, 0, 0, 0, -0.3],
       [True, '', False, 0, 'zscore', False, 0, 0, 0, 0, 0, 0, -0.3]
       ,[True, '', False, 0, 'zscore', False, 0, 0, 0, 0, 0, 0, -0.3]
       ,[True, '', False, 0, 'zscore', False, 0, 0, 0, 0, 0, 0, -0.3]]
#pool = multiprocessing.Pool(processes=3)
for i in range (0, len(var)):
    
    print(datetime.now())
    print('**************************************************************')
    print(int(flag/89 + 1), flagn, var[i])
    print('**************************************************************')

    try:

        if diff_param:
            v_list = df_res[var[i]]
            if param[i][0]:
                v_list = standardize(v_list, method = param[i][1])
            if param[i][2]:
                v_list = adding_random(v_list, size = param[i][3])
            df_res[var[i]] = v_list
            ic, longret, returns, pnl= plot_eval(df_res, var[i], weight = param[i][4], ifrank = True, complex_ver = param[i][5], nan = param[i][6], start_date = param[i][7], end_date = param[i][8], trimP=param[i][9], trimN=param[i][10], winsorP=param[i][11], winsorN=param[i][12])
            tmp = pd.DataFrame({'var': [var[i]], 'ic': ic, 'longret':longret, 'returns': returns})
            tmp.to_excel(writer, index = False, sheet_name = 'summary', startcol = 0, startrow = flag + 0)
            worksheet = writer.sheets['summary']
            worksheet.insert_image('F' + str(flag + 2), var[i] + '_eval.png')
            flag = flag + 89

        else:
            v_list = df_res[var[i]]
            v_list = standardize(v_list, method = '')
            #v_list = adding_random(v_list, size = 100)s
            df_res[var[i]] = v_list
            ic, longret, returns, pnl= plot_eval(df_res, var[i], ret = 'return', weight = 'rank', ifrank = True, complex_ver = False, nan = 0, start_date = 0, end_date = 0, trimP=0, trimN=0, winsorP=0, winsorN=0)

            tmp = pd.DataFrame({'var': [var[i]], 'ic': ic, 'longret':longret, 'returns': returns})
            tmp.to_excel(writer, index = False, sheet_name = 'summary', startcol = 0, startrow = flag + 0)
            worksheet = writer.sheets['summary']
            worksheet.insert_image('F' + str(flag + 2), var[i] + '_eval.png')
            flag = flag + 89

    except:
        print("*********************error**********************")
        err.append(var[i])
        print(int(flag/13), flagn, var[i])
        print('**************************************************************')
        flag = flag + 89

        
writer.save()

