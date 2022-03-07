import pandas as pd
import numpy  as np

from pprint import pprint
import json

datasets_path = 'dados_lcqar_07_08_2020'

co_df   = pd.read_csv('{}/CO.CSV'.format(datasets_path))
no2_df  = pd.read_csv('{}/NO2.CSV'.format(datasets_path))
o3_df   = pd.read_csv('{}/O3.CSV'.format(datasets_path))
so2_df  = pd.read_csv('{}/SO2.CSV'.format(datasets_path))
temp_df = pd.read_csv('{}/TMP.CSV'.format(datasets_path))
humi_df = pd.read_csv('{}/RH.CSV'.format(datasets_path))

gas_obj = {
    'CO':  co_df,
    'NO2': no2_df,
    'O3':  o3_df,
    'SO2': so2_df
}

climatic_element_obj = {
    'TEMP [ºC]': temp_df,
    'RH [%]':    humi_df
}

output_dict = {}
for gas in gas_obj:
    gas_df  = gas_obj[gas]
    gas_var = gas_df['Value'].var()

    statistics = gas_df['Value'].describe()
    statistics = statistics.to_dict()
    statistics['var'] = "%.3f" % gas_var

    output_dict[gas] = statistics

pprint(output_dict)

file_name = 'statistcs.json'
with open(file_name, 'w') as fp:
    json.dump(output_dict, fp)