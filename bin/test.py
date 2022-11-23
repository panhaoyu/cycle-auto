#!/usr/bin/env python3
# coding=utf-8
import pandas as pd

a = pd.read_csv('./cache/pnl/pnl.txt', header=None,sep='\t')
a = a.set_index(0)[4].to_frame('ret')
a['year'] = [i//10000 for i in a.index]


b = a.groupby('year').agg(
    ret_annual = ('ret', lambda x:x.mean() * 242),
    sharpe = ('ret', lambda x:x.mean()/x.std() * 15.5),
    mdd = ('ret', lambda x: ((1+x.cumsum().cummax())/(1+x.cumsum()) - 1).max() ),
)
print(b)
