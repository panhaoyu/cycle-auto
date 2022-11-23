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
        self.clse = self.pydpi.load_daily_data_file('clse')
        self.vol_in = self.pydpi.load_daily_data_file('volume_in')
        self.vol_out = self.pydpi.load_daily_data_file('volume_out')
        self.tkridx = self.pydpi.get_cidx()
        

        '''
        load necessary config from xml
        '''
        modNode = cfg.getChild("Config");
        ndayX = modNode.getAttrDefault("nday", 10);
        logger.info(f"running with config==>ndayX:{ndayX}")
        logger.info("done with initialization")

        '''
        user define variables
        '''
        self.ema_alpha = 2 / (ndayX + 1)
        self.vol_mean = np.zeros(len(self.tkrs))
        self.vol_square_mean = np.zeros(len(self.tkrs))
        self.price_mean = np.zeros(len(self.tkrs))
        self.price_square_mean = np.zeros(len(self.tkrs))
        self.pv_mean = np.zeros(len(self.tkrs))
        self.wts = np.zeros(len(self.tkrs))

    def push_interval_data(self, didx, tidx):
        date = self.pydpi.daily_time_series[didx][:-6]
        t = self.pydpi.time_series[tidx]
        ema_alpha = self.ema_alpha
        logger.info(f"Date:{date}, MinTime:{t}")

        vol = self.vol_out[didx, :] + self.vol_in[didx, :]
        price = self.clse[didx, :]
        '''
        for simplicity directly fillna here (Incorrect for alpha)
        '''
        vol = np.where(np.isnan(vol), 0, vol)
        price = np.where(np.isnan(price), 0, price)

        self.vol_mean = self.vol_mean * (1 - ema_alpha) + vol * ema_alpha
        self.vol_square_mean = self.vol_square_mean * (1 - ema_alpha) + (vol - self.vol_mean) ** 2 * ema_alpha
        self.price_mean = self.price_mean * (1 - ema_alpha) + price * ema_alpha
        self.price_square_mean = self.price_square_mean * (1 - ema_alpha) + (price - self.price_mean) ** 2 * ema_alpha
        self.pv_mean = self.pv_mean * (1 - ema_alpha) + (vol - self.vol_mean) * (price - self.price_mean) * ema_alpha

        '''
        for simplicity directly fillna here (Incorrect for alpha)
        '''
        self.vol_mean = np.where(np.isnan(self.vol_mean), 0, self.vol_mean)
        self.vol_square_mean = np.where(np.isnan(self.vol_square_mean), 0, self.vol_square_mean)
        self.price_mean = np.where(np.isnan(self.price_mean), 0, self.price_mean)
        self.price_square_mean = np.where(np.isnan(self.price_square_mean), 0, self.price_square_mean)
        self.pv_mean = np.where(np.isnan(self.pv_mean), 0, self.pv_mean)

        wts = self.pv_mean / np.sqrt(self.vol_square_mean * self.price_square_mean)
        wts = -(wts - np.nanmean(wts)) / np.nanstd(wts)
        self.wts = wts # wts is float pointer, do not perform any operator directly on wts
        debug = True
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

