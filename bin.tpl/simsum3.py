#!/usr/bin/env python3
# coding=utf-8
import os 
import numpy as np
import pandas as pd
import click
#pd.set_option('max_columns', None)
pd.set_option('display.expand_frame_repr', False)
NUM_DAYS = 242

def runSim(filepath, dtype='yearly', starttime='093900', endtime='150000',
        startdate='0', enddate='0'):
    output = ""
    assert os.path.exists(filepath), f"{filepath} does not exist"
    df = pd.read_csv(filepath)
    df['dateStr'] = df['dateStr'].astype('str')
    df['tme'] = df['tme'].apply(lambda x:'0'*(6-len(str(x))) + str(x))
    df['month'] = df['dateStr'].apply(lambda x:int(x[:6]))
    df['year'] = df['dateStr'].apply(lambda x:int(x[:4])) 
    df['pnl'] *= (df['long'] - df['short']) / (df['long']) 
    df['trials'] = 1 

    # filter dataframe
    if startdate != "0":
        df = df[df.dateStr >= startdate]
    if enddate != "0":
        df = df[df.dateStr <= enddate]

    df = df[df.tme >= starttime]
    df = df[df.tme <= endtime]

    df = df.groupby('dateStr').mean()
    for col in ['year', 'month']:
        df[col] = df[col].apply(lambda x:str(int(x)))

    if dtype == 'yearly':
        group_cols = 'year'
    elif dtype == 'monthly':
        group_cols = 'month'
    else:
        raise AssertionError(f"dtype:{dtype} does not support! choose from yearly and monthly")

    def get_mdd_start_date(x):
        x_cumsum = x.cumsum()
        end_idx = (x_cumsum.cummax() - x_cumsum).idxmax()
        return x_cumsum.cummax().loc[:end_idx].pipe(lambda x:x[x.diff()!=0]).index[-1]

    def groupby(df, group_cols):
        dftmp = df.groupby(group_cols).agg(
            begin = ('pnl', lambda x:x.index[0]),
            end = ('pnl', lambda x:x.index[-1]),
            long = ('long', lambda x:x.mean()),
            short = ('short', lambda x:x.mean()),
            ret = ('pnl', lambda x:x.mean() * 100 * NUM_DAYS),
            tvr = ('turnover', lambda x:x.mean() * 100),
            sharpe = ('pnl', lambda x:x.mean()/x.std() * np.sqrt(NUM_DAYS)),
            drawdown = ('pnl', lambda x:(x.cumsum().cummax() - x.cumsum()).max() * 100),
            win_rate = ('pnl', lambda x:(x > 0).sum() / x.shape[0] * 100),
            longNum = ('longNum', lambda x:int(x.mean())),
            shortNum = ('shortNum', lambda x:int(x.mean())),
            dd_start = ('pnl', get_mdd_start_date),
            dd_end = ('pnl', lambda x:(x.cumsum().cummax() - x.cumsum()).idxmax()),
            days = ('pnl', lambda x:x.shape[0])
        ) 
        dftmp['bp_mrgn'] =  dftmp.ret / dftmp.tvr / NUM_DAYS  * 1e4
        return dftmp[['begin','end','long','short','ret','tvr','sharpe','drawdown','win_rate','bp_mrgn','longNum','shortNum','dd_start','dd_end','days']]
    stats = groupby(df, group_cols)
    
    output += (f'==========={filepath.split("/")[-1].split(".")[0]}==========')
    output += '\n'
    output+=("%17s %7s %7s %6s %7s %6s %5s %6s %6s %5s %5s %8s %8s %5s" % \
          ("dates", "long", "short", "%ret",  "%tvr", "sharpe","%dd","%win", "margin", "lnum", "snum", 
            "dd_st", "dd_et",'tdays'))
    output += '\n'
    for group, values in stats.iterrows():
        output+=("{:8s}-{:8s} {:7.2f} {:7.2f} {:6.2f} {:7.2f} {:6.2f} {:5.2f} {:6.2f} {:6.2f} {:5d} {:5d} {:8s} {:8s} {:5d}".format(*list(values)))
        output += '\n'

    summary = groupby(df, 'trials')  
    summary['bp_mrgn'] = summary.ret / summary.tvr / NUM_DAYS  * 1e4
    output += '\n'
    for group, values in summary.iterrows():
        output+=("{:8s}-{:8s} {:7.2f} {:7.2f} {:6.2f} {:7.2f} {:6.2f} {:5.2f} {:6.2f} {:6.2f} {:5d} {:5d} {:8s} {:8s} {:5d}".format(*list(values)))
    return output



if __name__ == "__main__":
    # output = runSim(filepath=os.path.expanduser('~/pnls/Alpha_ZY00051_4.pnl.txt'))
    output = runSim(filepath=os.path.expanduser('~/work/alpha/pybsim/bin/cache/pnl/pnl.txt'))
    #print(output)

    




