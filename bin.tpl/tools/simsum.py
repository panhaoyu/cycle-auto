#!/usr/bin/env python

import sys,pdb
from optparse import OptionParser
import numpy as np

cols = 8
days = 240
pnlfile = sys.argv[1]

parser = OptionParser()
parser.add_option('-t', action='store', type='string', default='yearly', dest='timeframe')
parser.add_option('-s', action='store', type='int', default='0', dest='startdate')
(options, sys.argv) = parser.parse_args()

monthly = False
if (options.timeframe == 'monthly') :
    monthly = True

rawlines = file(pnlfile).readlines()
if options.startdate > 0 :
    lines = []
    for line in rawlines :
        if int(line[0:8]) >= options.startdate : 
            lines.append(line)
else :
    lines = rawlines
dates = np.empty(len(lines), dtype=np.int)
pnl = np.empty((cols, len(lines)), dtype=np.float32)

def read_lines():
    keydates = []
    last = 0
    for d in xrange(len(lines)) :
        item = lines[d].split()
        dates[d] = int(item[0])
        if d == 0 or (monthly and dates[d] / 100 != dates[d - 1] / 100) or (dates[d] / 10000 != dates[d - 1] / 10000) :
            keydates.append([dates[d], 0, d])
            if d > 0 :
                keydates[-2][1] = dates[d - 1]
        for i in xrange(cols) :
            pnl[i][d] = float(item[i + 1])
    keydates[-1][1] = dates[-1]
    return keydates

def calc(d0, d1):
    for d in range(d0, d1) :
        if pnl[1][d] != 0. or pnl[2][d] != 0. :
            d0 = d
            break

    tpnl = np.sum(pnl[0][d0:d1])
    long = np.mean(pnl[1][d0:d1]) / 1e6
    short = np.mean(pnl[2][d0:d1]) / 1e6
    ret = np.mean(pnl[3][d0:d1])
    val_hld = np.sum(pnl[4][d0:d1])
    val_trd = np.sum(pnl[5][d0:d1])
    sh_trd = np.sum(pnl[7][d0:d1])
    dwin = np.sum(pnl[0][d0:d1] > 0)

    tvr = val_trd / val_hld
    ir = ret / np.std(pnl[3][d0:d1], ddof=1) * np.sqrt(days)
    bp_mgn = tpnl / val_trd * 10000
    cs_mgn = tpnl / sh_trd * 100
    ret *= days
    pwin = np.float32(dwin) / (d1 - d0 + 1.)

    dd = 0.; high = 0.; cpnl = 0.;
    ds = d0; dh = d0; de = d0;
    for d in range(d0, d1) :
        cpnl += pnl[0][d]
        if cpnl > high :
            dh = d
            high = cpnl
        cdd = cpnl - high
        if cdd < dd :
            ds = dh; de = d
            dd = cdd
    dd /= -(long * 1e4)
    return (long, short, ret, tvr, ir, bp_mgn, cs_mgn, dd, ds, de, pwin)

def main ():
    keydates = read_lines()
    print ('%10s%10s%8s%8s%10s%10s%10s%10s%10s%10s%10s%10s') \
        % ('from', 'to', 'long', 'short', 'return', 'tvr', \
            'sharpe', 'drawdown', 'dd_start', 'dd_end', 'bp_mrgn', 'winrate')
    for i in xrange(len(keydates)) :
        d0 = keydates[i][2]
        d1 = len(dates)
        if i < len(keydates) - 1:
            d1 = keydates[i + 1][2]
        (long, short, ret, tvr, ir, bp_mgn, cs_mgn, dd, ds, de, pwin) = calc(d0, d1)
        print ('%10d%10d%8.2f%8.2f%10.5f%10.5f%10.5f%10.5f%10d%10d%10.5f%10.5f') \
            % (keydates[i][0], keydates[i][1], long, short, ret, tvr, ir, \
                dd, dates[ds], dates[de], bp_mgn, pwin)
    (long, short, ret, tvr, ir, bp_mgn, cs_mgn, dd, ds, de, pwin) = calc(0, len(dates))
    print
    print ('%10d%10d%8.2f%8.2f%10.5f%10.5f%10.5f%10.5f%10d%10d%10.5f%10.5f') \
        % (keydates[0][0], keydates[-1][1], long, short, ret, tvr, ir, \
            dd, dates[ds], dates[de], bp_mgn, pwin)

if __name__ == '__main__':
    main()

