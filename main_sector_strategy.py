import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from datetime import datetime


def get_from_db(countries,start,end):
    from backtest_engine.database.borsdata.borsdata_price_database_script import priceDataBase
    from backtest_engine.database.borsdata.borsdata_kpi_database_script import kpiDataBase

    from backtest_engine.utils import merge_kpi_price
    
    tickers = []
    dfs = {}
    for country, markets in countries.items():
        for market in markets:
            db_price = priceDataBase(country, market, start, end)
            db_kpi = kpiDataBase(61, market, country)
            price_tickers, price_dfs = db_price.export_database()
            kpi_tickers,kpi_dfs_dfs = db_kpi.export_database()
            tickers0,dfs0 = merge_kpi_price(
                                        kpi_dfs_dfs,
                                        price_dfs,
                                        kpi_tickers,
                                        price_tickers,
                                        'aktier'
                                    )
            
            tickers += tickers0
            dfs |= dfs0
            del tickers0
            del dfs0
    return tickers,dfs
        
                                                   
def get_ticker_dfs(countries,start,end):
    from backtest_engine.utils import load_pickle,save_pickle
    try:
        tickers, dfs = load_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_sector_strat.obj")
    except Exception as err:
        tickers,dfs = get_from_db(countries,start,end)
        save_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_sector_strat.obj", (tickers,dfs))
    return tickers, dfs

countries = {'Sverige':['Large Cap', 'Mid Cap'],
             'Norge':['Oslo Bors', 'Oslo Expand'],
             'Finland':['Large Cap', 'Mid Cap'],
             'Danmark':['Large Cap', 'Mid Cap']
             }
start = '2014-01-01'
end = '2023-09-01'

tickers, dfs = get_ticker_dfs(countries=countries,start=start,end=end)

information1 = pd.read_csv('/Users/oskarfransson/vs_code/trading/backtest_engine/database/borsdata/instrument_with_meta_data.csv')
information2 = information1[
    (information1['country'] == 'Sverige') | 
    (information1['country'] == 'Norge') | 
    (information1['country'] == 'Danmark') |
    (information1['country'] == 'Finland')
    ]

arr_of_sectors = information2['sector'].dropna().unique()
sector_map = {}
for sector in arr_of_sectors:
    tickers1 = information2[information2['sector'] == sector]
    tickers2 = tickers1[(tickers1['market'] == 'Large Cap') 
                        | (tickers1['market'] == 'Mid Cap') 
                        | (tickers1['market'] == 'Oslo Bors') 
                        | (tickers1['market'] == 'Oslo Expand')]
    sector_map[sector] = tickers2.name.unique().tolist()

closes = {}
for ticker in tickers:
    df = dfs[ticker].resample('1w').last()
    closes[ticker] = df['close']

closes_df = pd.concat(closes,axis=1)

sector_rets = {}
for sector,names_list in sector_map.items():
    names_list = [symb.replace(' ','_') for symb in names_list]
    names_list = [symb.replace('&','and') for symb in names_list]
    names_list = [symb.replace('-','_') for symb in names_list]
    names_list = [symb.replace('.','') for symb in names_list]

    tmp = []
    for n in names_list:
        for j in closes_df.columns.tolist():
            if n == j:
                tmp.append(n)

    tmp_unique = pd.Series(tmp).unique().tolist()
    sector_rets[sector] = closes_df[tmp_unique].pct_change()

# from sklearn.decomposition import PCA
# sector_index = {}
# for sect in list(sector_map.keys()):
#     pca = PCA(n_components=1)
#     data = sector_rets[sect].fillna(0)
#     pca.fit(data.loc[:'2017-12-31'])
#     zs = pca.transform(data)
#     sector_index[sect] = (1+pd.Series(data=np.dot(zs,pca.explained_variance_ratio_),index=data.index)).cumprod()

#     sector_index[sect].plot()
# plt.show()

sector_index = {}
plt.figure(figsize=(12,5))
for sector in sector_rets.keys():
   rets = sector_rets[sector].copy()
   scaled_rets = rets/(rets.ewm(0.94).std()*np.sqrt(52))
   m = (1+scaled_rets.mean(axis=1)).cumprod()
   sector_index[sector] = scaled_rets.mean(axis=1)
   plt.plot(m.apply(np.log),label=sector)

plt.legend()
plt.close()

correlations = pd.concat(sector_index, axis=1).loc[:'2017-12-31'].corr(method='spearman')
plt.figure(figsize=(12,5))
dissimilarity = 1 - abs(correlations)
Z = linkage(squareform(dissimilarity), 'complete')

dendrogram(Z, labels=pd.concat(sector_index, axis=1).columns, orientation='top', 
           leaf_rotation=90)

plt.tight_layout()
plt.close()

cluster1 = ['Industri',
           'Finans & Fastighet',
           'Sällanköpsvaror', 
           'Material',
           'Dagligvaror'
        ]
cluster2 = ['Energi', 'Kraftförsörjning']
cluster3 = ['Telekommunikation']
cluster4 = ['Hälsovård','Informationsteknik']

from backtest_engine.strategies.sector_strategy import SectorAlpha

sectors_dict = {}
for sector in sector_map.keys():
    names_list = sector_map[sector]
    names_list = [symb.replace(' ','_') for symb in names_list]
    names_list = [symb.replace('&','and') for symb in names_list]
    names_list = [symb.replace('-','_') for symb in names_list]
    names_list = [symb.replace('.','') for symb in names_list]

    tmp = []
    for n in names_list:
        for j in tickers:
            if n == j:
                tmp.append(n)

    tmp_unique = pd.Series(tmp).unique().tolist()
    sectors_dict[sector] = {ticker:dfs[ticker] for ticker in tmp_unique}

cluster1_dict = {}
for i in range(len(cluster1)):
    cluster1_dict0 = sectors_dict[cluster1[i]].copy()
    cluster1_dict |= cluster1_dict0

cluster2_dict = {}
for i in range(len(cluster2)):
    cluster2_dict0 = sectors_dict[cluster2[i]].copy()
    cluster2_dict |= cluster2_dict0

cluster3_dict = {}
for i in range(len(cluster3)):
    cluster3_dict0 = sectors_dict[cluster3[i]].copy()
    cluster3_dict |= cluster3_dict0

cluster4_dict = {}
for i in range(len(cluster4)):
    cluster4_dict0 = sectors_dict[cluster4[i]].copy()
    cluster4_dict |= cluster4_dict0

period_start = datetime(2013,1,1)
period_end = datetime(2023,9,1)

alpha1 = SectorAlpha(insts=list(cluster1_dict.keys()),
                     dfs=cluster1_dict,
                     start=period_start,
                     end=period_end,
                     trade_frequency='monthly')
df1 = alpha1.run_simulation(use_vol_target=False)

alpha2 = SectorAlpha(insts=list(cluster2_dict.keys()),
                     dfs=cluster2_dict,
                     start=period_start,
                     end=period_end,
                     trade_frequency='monthly')
df2 = alpha2.run_simulation(use_vol_target=False)

alpha3 = SectorAlpha(insts=list(cluster3_dict.keys()),
                     dfs=cluster3_dict,
                     start=period_start,
                     end=period_end,
                     trade_frequency='monthly')
df3 = alpha3.run_simulation(use_vol_target=False)

alpha4 = SectorAlpha(insts=list(cluster4_dict.keys()),
                     dfs=cluster4_dict,
                     start=period_start,
                     end=period_end,
                     trade_frequency='monthly'
                     )
df4 = alpha4.run_simulation(use_vol_target=False)

plt.figure(figsize=(12,5))
df1['capital'].plot()
df2['capital'].plot()
df3['capital'].plot()
df4['capital'].plot()
plt.show()

sector_momentums = pd.concat([alpha1.mom, 
                              alpha2.mom,
                              alpha3.mom, 
                              alpha4.mom,
                            ],axis=1)
sector_rets = pd.concat([df1['capital_ret'], 
                         df2['capital_ret'], 
                         df3['capital_ret'],
                         df4['capital_ret']
                        ],axis=1)
eom = pd.concat([alpha1.trading_day_ser, 
                 alpha2.trading_day_ser,
                 alpha3.trading_day_ser,
                 alpha4.trading_day_ser,
                 ],axis=1)

sector_momentums.columns = ['sector1','sector2','sector3','sector4']
sector_rets.columns = ['sector1','sector2','sector3','sector4']
eom.columns = ['sector1','sector2','sector3','sector4']
n_sectors = 1
ranking = (sector_momentums.rank(ascending=True,axis=1) > (4-n_sectors))
ranking = pd.DataFrame(np.where(eom, ranking, np.NaN),index=ranking.index,columns=ranking.columns).fillna(method='ffill')
strat = (1+(ranking*sector_rets).sum(axis=1)/n_sectors).cumprod()
qs.reports.html(strat,'SPY',output='/Users/oskarfransson/vs_code/trading/sector_strategy.html')
