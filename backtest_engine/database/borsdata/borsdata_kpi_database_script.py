import pandas as pd
import sqlalchemy
import threading

from sqlalchemy import text
from sqlalchemy import inspect

from backtest_engine.database.borsdata.constants import API_KEY
from backtest_engine.database.borsdata.borsdata_client import BorsdataClient
from backtest_engine.database.borsdata.borsdata_api import BorsdataAPI

class kpiDataBase:
    def __init__(self, kpi_val, market, country):
        self.api = BorsdataAPI(API_KEY)
        self.client = BorsdataClient()
        
        kpi_id_map = pd.read_csv('/Users/oskarfransson/vs_code/trading/backtest_engine/database/borsdata/kpi_ids.csv',sep = ';')[['Name','KpiId','Description']]
        self.kpi_id_map = kpi_id_map[(kpi_id_map.Description == 'R12 Mean') | (kpi_id_map.Description == 'R12')].drop(columns = 'Description')
       
        def _set_vars(val):
            if isinstance(val,str):
                self.kpi_val = self.kpi_id_map[self.kpi_id_map.Name == val].KpiId
            
            else:
                self.kpi_val = val
            
            self.kpi_name = self.kpi_id_map[self.kpi_id_map.KpiId == self.kpi_val].values.tolist()[0][0]
            self.kpi_name = self.kpi_name.replace("/", "_")
            
            self.market = market
            self.country = country
            # self.name_id_map = pd.read_csv('/Users/oskarfransson/vs_code/trading/database/borsdata/instrument_with_meta_data.csv')[['name','ins_id']]
            self.csv_file = self.client.instruments_with_meta_data()
            self.name_id_map = self.csv_file[['name','ins_id']]
            
            engine_name = f'{self.kpi_name}_{self.market}_{self.country}'
            self.engine_name = engine_name.replace(" ", "_")
            self.engine = sqlalchemy.create_engine('sqlite:////Users/oskarfransson/vs_code/trading/backtest_engine/database/files/' + self.engine_name)
            
            return None
        
        try:
            if kpi_val is None:
                print('Choose an id from list: ')
                print(self.kpi_id_map)
                selection = input('Enter id: ')
                _set_vars(int(selection))
            
            else:
                _set_vars(kpi_val)
                
        except ValueError:
                print(f'{kpi_val} dont exist in API')
                print('Choose valid id from list: ')
                print(self.kpi_id_map)
                selection = input('Enter id: ')
                _set_vars(int(selection))
                         
    def create_database(self):
        kpi_data = pd.concat(self.client.history_kpi(self.kpi_val, self.market, self.country))
        names = self.client.names

        report_date_df = {}
        data_list = []
        names_list = []

        def _helper(name_value):
            n = kpi_data[kpi_data.name == name_value].shape[0]
            ins_id = self.name_id_map[self.name_id_map.name == name_value].ins_id.values.tolist()[0]
            report_date_df[name_value] = self.api.get_instrument_report(int(ins_id), 'r12', n).reportDate

        threads = [threading.Thread(target=_helper, args=(name_value,)) for name_value in names]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]

        for name_value in names:
            df0 = kpi_data[kpi_data.name == name_value]
            reports_date = report_date_df[name_value]
            idx = pd.to_datetime(df0.year.astype(str) + 'Q' + df0.period.astype(str))
            kpis = pd.Series(df0.kpiValue.values, index=idx, name = df0.name[0])
            reports_date.index = idx
            data_frame = pd.concat([kpis, reports_date], axis=1)
            name_value = name_value.replace("&", "and")
            name_value = name_value.replace(" ", "_")
            name_value = name_value.replace(".", "")
            name_value = name_value.replace("-", "_")
            data_frame.columns = [name_value,f'{name_value}_reportDate']

            data_list.append(data_frame)
            names_list.append(name_value)

        data = pd.concat(data_list, axis = 1)
        
        for name in names_list:
            try:
                new_data = data[[name,f'{name}_reportDate']]
                d = pd.to_datetime(new_data[f'{name}_reportDate'].values) + pd.offsets.QuarterBegin(1)
                temp = new_data.reset_index().copy()
                end_stamp = d.dropna()[0]

                counter = False
                i = 0
                while (not counter) or (i > d.shape[0]):
                    if d[i]==end_stamp:
                        counter = True
                    else:
                        i += 1

                date_range1 = pd.date_range(end=end_stamp,periods=i+1, freq='Q') + pd.offsets.QuarterBegin(0)
                date_range2 = pd.date_range(start=end_stamp, periods=d.shape[0] - i - 1, freq='Q') + pd.offsets.QuarterBegin(0)
                idx = date_range1.union(date_range2)
                temp['datetime'] = idx
                new_data2 = temp.set_index('datetime').drop(columns='index')
                
                new_data2.to_sql(name, self.engine)
                print(f'Imported {name}')
            
            except IndexError:
                try:
                    new_data = data[[name,f'{name}_reportDate']]
                    new_data2 = new_data.reset_index(inplace=True)
                    new_data2 = new_data2.rename(columns={'index':'datetime'})
                    new_data2 = new_data2.set_index('datetime')
                    new_data2.to_sql(name, self.engine)
                except:
                    print(f'Skipped {name}')
                    pass
            
        return None   
    
    def export_database(self):
        data_frames = {}
        inspector = inspect(self.engine)
        symbols = inspector.get_table_names()
        with self.engine.begin() as conn:
            for symbol in symbols:
                query = text(f'SELECT * FROM "{symbol}"')
                df = pd.read_sql_query(query, conn)#.set_index('new_index')
                data_frames[symbol] = df
        
        return symbols, data_frames
    
        