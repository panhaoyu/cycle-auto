#codeing:utf-8
import sys
sys.path.append("/datas/share/Msimrelease_stkbar/lib/")
from pybase_model import hfs_base_model
from pyxmlparser import *
import numpy as np
from hfs_data_api_py import * 
from pydata_api import *
from pydata import *
#import Oputil

class PYMODEL(hfs_base_model):
    def __init__(self, cfg):
        super(PYMODEL ,self).__init__(cfg)
        print('call PYMODEL.__init__')
        print('tkrsize:', len(self.tkrs))

        cfg_root = cfg.getRoot()
        print(f"DATA:{cfg_root.getChild('Macros').getAttrDefault('DATA','')}")
        DATA = cfg_root.getChild('Macros').getAttrDefault('DATA','')
        self.pydpi = hfs_data_api_py(DATA) 
        self.clse = self.pydpi.load_data_file('clse')
        self.tkridx = self.pydpi.get_cidx()
        

        #interface from C++ 
        self.clse2 = self.dpi.get_data_float('clse')
        self.tkridx2 = self.dpi.get_cidx()

        modNode = cfg.getChild("Config");
        ndayX = modNode.getAttrDefault("nday", 10);
        print(ndayX)

        for uidx in np.arange( len(self.tkridx) ):
            print("uidx=", uidx, "  cidx=", self.tkridx[uidx], " cidx2=", self.tkridx2[uidx])
        #self.tkrs
        #self.univ


    def push_interval_data(self, didx, tidx):
        #print("call push_interval_data", didx, tidx)
        print('clse', self.clse[didx, tidx, 0])
        print('clse2', self.clse2[didx, tidx, 0])
        self.wts = np.arange(len(self.tkrs))
        #for uidx in np.arange( len(self.cidx) ):
        #    print("uidx=", uidx, "  cidx=", self.cidx[uidx])
        return True

    def __del__(self):
        print("call __del__")

    ##def get_forecasts(self):
    ##    print('call get_forecasts')
    ##    return self.wts


def create(cfg):
    return PYMODEL(cfg)

