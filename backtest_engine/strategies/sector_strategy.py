import numpy as np
import pandas as pd

from backtest_engine.utils import BacktestEngine

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


class SectorAlpha(BacktestEngine):

    def __init__(self,insts,dfs,start,end,trade_frequency):
        super().__init__(insts,dfs,start,end,trade_frequency=trade_frequency)
    
    def pre_compute(self,trade_range):
        self.op4s = {}
        for inst in self.insts:
            inst_df = self.dfs[inst]

            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3       
            self.op4s[inst] = op4
        return 
    
    def post_compute(self,trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst]["op4"] = self.op4s[inst]
            temp.append(self.dfs[inst]["op4"])

        temp_df = pd.concat(temp,axis=1)
        temp_df.columns = self.insts
        temp_df = temp_df.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        zscore = lambda x: (x - np.nanmean(x))/np.nanstd(x)
        cszcre_df = temp_df.fillna(method="ffill").apply(zscore, axis=1, raw=True)
        
        alphas = []
        cl_dict = {}
        momentum_dict = {}
        for inst in self.insts:
            df = self.dfs[inst]["close"].copy()
            self.dfs[inst]["alpha"] = cszcre_df[inst].rolling(35).mean() * -1
            
            alphas.append(self.dfs[inst]["alpha"])
            cl_dict[inst] = df / df.rolling(253).mean() - 1
            momentum_dict[inst] = (df - df.shift(253)) > 0.

        alphadf = pd.concat(alphas,axis=1)
        alphadf.columns = self.insts
        momentum_filter = pd.concat(momentum_dict,axis=1)
        self.mom = pd.concat(cl_dict,axis=1).mean(axis=1)

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf)) & (momentum_filter)
        self.alphadf = alphadf
        masked_df = self.alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        num_eligibles = self.eligiblesdf.sum(axis=1)
        rankdf= masked_df.rank(axis=1,method="average",na_option="keep",ascending=True)
        numb = num_eligibles - 10
        longdf = rankdf.apply(lambda col: col > numb, axis=0, raw=True)
        forecast_df = longdf.astype(np.int32)
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts