import pandas as pd
from collections import OrderedDict
import os
import errno
import argparse
from datetime import datetime

# from BsimPy.lib import (mkdir, cd, getCurrentTime)

def getCurrentTime():
    """
    """
    return datetime.now().strftime('%H:%M:%S') 

def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def createSummary(factorStatsDir, statsFile):
    """
    """
    print getCurrentTime(), "Putting together factors' stats..."

    paths = (os.path.join(root, filename) for root, _, filenames in os.walk(factorStatsDir) for filename in filenames)
    statsPath = [p for p in paths if p.endswith('.csv') and "summary" in p]

    fDir = os.path.join(factorStatsDir, 'summary')
    mkdir(fDir)

    allFactors = [pd.read_csv(x) for x in statsPath]

    allFactors = pd.concat(allFactors, copy=False, ignore_index=True)

    # allFactors.to_csv(os.path.join(factorStatsDir, "alphaStats.csv"))
    allFactors.to_csv(statsFile)




def factorPairCorr(factorPnl):
    """
    """
    print getCurrentTime(), "Calculating factors' pairwise correlation..."
    if isinstance(factorPnl, pd.DataFrame):
        return factorPnl.corr()



def getFactorsPnl(factorStatsDir):
    """
    concat all factors' pnl
    """
    print getCurrentTime(), "Loading factors' pnl..."

    paths = (os.path.join(root, filename) for root, _, filenames in os.walk(factorStatsDir) for filename in filenames)
    pnlPath = [p for p in paths if p.endswith('.csv') and not "summary" in p]
    statsPath = [p for p in paths if p.endswith('.csv') and "summary" in p]

    factorNames = (x.split('/')[-1][:-4] for x in pnlPath )
    # replace "," with "~" 
    factorNames = map(lambda x: x.replace(",", "~"), factorNames)
    

    pnls  = pd.concat((pd.read_csv(f, index_col=[0], usecols=['time', 'LS']) for f in pnlPath), axis=1)
    pnls.columns = factorNames

    return pnls

def getAllStats(path):
    """
    """
    statsAll = pd.read_csv(path, index_col=[1])
    statsAll.drop('Unnamed: 0', axis=1, inplace=True)
    statsAll.sort_values(by='lssharpe', ascending=False, inplace=True)
    statsAll['abs(ic)'] = statsAll['ic'].abs()
    # statsAll['abs(IR)'] = statsAll['IR'].abs()

    return statsAll


def filterFactors(factorStatsDir, filters=None):
    """
    """
    def corrFilter(statsAll, facCorr, threshhold=0.7):
        index = statsAll.index.values

        valid = [index[0]]
        maxCorrs = [0]
        for i, idx in enumerate(index[1:]):
            corrVal = facCorr.loc[idx, valid]
            maxCorr = corrVal.max()

            if  maxCorr < threshhold:
                valid.append(idx)
                maxCorrs.append(maxCorr)

        return valid, maxCorrs



    facCorr = factorPairCorr(getFactorsPnl(factorStatsDir))
    path = os.path.join(factorStatsDir, "summary/alphaStats.csv")

    if not os.path.isfile(path):
        createSummary(factorStatsDir, path)
        

    statsAll = getAllStats(path)

    # filters
    statsAll = statsAll[statsAll.lssharpe > 2]

    valid, maxCorrs = corrFilter(statsAll, facCorr)

    validFactors = statsAll.loc[valid]
    validFactors.to_csv(os.path.join(factorStatsDir, "summary/validFactors.csv"))

    return validFactors



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', action='store', dest='factorStatsDir', default='',
                        help='Path to factorStats')
    results = parser.parse_args()

    factorStatsDir = results.factorStatsDir

    if factorStatsDir:
        s = filterFactors(factorStatsDir)
        print "Success"
    else:
        print "-p <factorStatsDir>"

    
