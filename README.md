# Backtest-Sys
Backtest system with code,csvs,example alpha
import_datas =['close','fore','ret_bwd','bm_ret_bwd','mktc','sector','segment']

for key in btd.database.keys():
    print(key)
    btd.database[key].iloc[:,:500].to_csv('./'+key+'.csv',)
