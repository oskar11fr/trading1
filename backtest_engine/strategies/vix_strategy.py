import numpy as np
import pandas as pd
from backtest_engine.utils import BacktestEngine

class VixStrategy(BacktestEngine):

    def __init__(self,insts,dfs,start,end,trade_frequency):
        super().__init__(insts,dfs,start,end,trade_frequency)
    
    def pre_compute(self,trade_range):
        vix = self.dfs['_VIX'].close
        vix3m = self.dfs['_VIX3M'].close
        ratio = ((vix3m/vix).fillna(method='ffill'))
        filtered_ratio = ratio/ratio.rolling(23).mean()
        del self.dfs['_VIX']
        del self.dfs['_VIX3M']
        self.insts = list(self.dfs.keys())
        self.dfs['filtered_ratio'] = filtered_ratio
        self.dfs['ratio'] = ratio
        return 
    
    def post_compute(self,trade_range):
        filtered_ratio = self.dfs['filtered_ratio'].copy()
        ratio = self.dfs['ratio'].copy()
        temp_alpha = []
        names = []
        eows = []
        for inst in self.insts:
            if inst != 'SVXY':
                self.dfs[inst]['ratio_cond'] = (filtered_ratio < 1).astype(int)
                names.append(inst)
                temp_alpha.append(self.dfs[inst]['ratio_cond'])
            if inst == 'SVXY':
                self.dfs[inst]['ratio_cond'] = (filtered_ratio > 1).astype(int)
                names.append(inst)
                temp_alpha.append(self.dfs[inst]['ratio_cond'])
        
        alpha_df = pd.concat(temp_alpha,axis=1)
        alpha_df.columns = names
        self.eligblesdf = self.eligiblesdf & (~pd.isna(alpha_df))
        self.forecast_df = alpha_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts