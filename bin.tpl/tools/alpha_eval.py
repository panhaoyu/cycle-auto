import glob
import os
import numpy as np
import pandas as pd

def alpha_eval(path, ann_factor=242, min_sharpe=2, min_ret=0.15, max_corr=0.7, max_turn=0.8):
    fnames = glob.glob(path + '/*/pnls.txt')
    stats = []
    rdf = None
    tidx = None
    for fname in fnames:
        df = pd.read_csv(fname, sep='\t')
        rets = df.long_short_nav.diff()
        if rdf is None:
            rdf = rets
            tidx = df.date.astype('str')
        else:
            rdf = pd.concat([rdf, rets], axis=1)
        winr = (rets > 0).mean()
        winr = max(winr, 1 - winr)
        aret = rets.mean() * ann_factor
        avol = rets.std() * np.sqrt(ann_factor)
        sharpe = aret / avol
        lturns = df.long_turnover.mean()
        sturns = df.short_turnover.mean()
        stats.append([np.sign(aret), abs(aret), avol, abs(sharpe), winr, lturns, sturns])

    anames = [fname.split(os.path.sep)[-2] for fname in fnames]
    rdf.columns = anames
    rdf.index.name = 'date'
    rdf.index = pd.to_datetime(tidx)
    rdf.to_csv('alpha_returns.csv')

    alpha_corr = rdf.corr()
    alpha_corr.index.name = 'alpha'
    alpha_corr.to_csv('alpha_correlations.csv')
    cols = ['sign', 'ann_ret', 'ann_vol', 'sharpe', 'win_rate', 'long_turns', 'short_turns']
    sdf = pd.DataFrame(stats, columns=cols, index=anames)
    sdf = sdf.sort_values('sharpe', ascending=False)
    sdf.index.name = 'alpha'
    sdf['sharpe_check'] = 'Failed'
    sdf['correl_check'] = 'Failed'
    sdf['annret_check'] = 'Failed'
    sdf['turns_check'] = 'Failed'
    sdf['final_status'] = 'Reject'
    pool = []
    for alpha in sdf.index:
        pass_count = 0
        if sdf.loc[alpha].sharpe > min_sharpe:
            sdf.loc[alpha, 'sharpe_check'] = 'Pass'
            pass_count += 1
        if sdf.loc[alpha].ann_ret > min_ret:
            sdf.loc[alpha, 'annret_check'] = 'Pass'
            pass_count += 1
        if sdf.loc[alpha].long_turns < max_turn:
            sdf.loc[alpha, 'turns_check'] = 'Pass'
            pass_count += 1
        for pa in pool:
            if abs(alpha_corr.loc[alpha, pa]) > max_corr:
                break
        else:
            sdf.loc[alpha, 'correl_check'] = 'Pass'
            pass_count += 1
        if pass_count == 4:
            sdf.loc[alpha, 'final_status'] = 'Accept'
            pool.append(alpha)
    sdf.to_csv('alpha_evals.csv')
    print(sdf.loc[pool])
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Alphas')
    parser.add_argument('-p', '--path', metavar='P', default='cache/alphas',
        help='Path to alphas, default=cache/alphas')
    parser.add_argument('-a', '--annual-factor', metavar='A', default=242, type=int,
        help='Annualize factor, default=242')
    parser.add_argument('-s' ,'--min-sharpe', metavar='S', default=2, type=float,
        help='Minimum sharpe required, default=2')
    parser.add_argument('-r', '--min-return', metavar='R', default=0.15, type=float,
        help='Minimum annualized return required, default=0.15')
    parser.add_argument('-c', '--correlation', metavar='C', default=0.7, type=float,
        help='Maximum correlation allowed in the alpha pool, default=0.7')
    parser.add_argument('-t', '--turnover', metavar='T', default=0.8, type=float,
        help='Maximum turnover allowed, default=0.8')
    args = parser.parse_args()
    alpha_eval(args.path, args.annual_factor, args.min_sharpe, args.min_return, args.correlation, args.turnover)
