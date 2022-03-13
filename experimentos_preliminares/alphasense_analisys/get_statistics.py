import pandas as pd
import numpy  as np

from pprint import pprint
import json

datasets_path = 'alphasense_data'

co_df   = pd.read_csv('{}/ISB_CO.CSV'.format(datasets_path))
h2s_df  = pd.read_csv('{}/ISB_H2S.CSV'.format(datasets_path))
no2_df  = pd.read_csv('{}/ISB_NO2.CSV'.format(datasets_path))
o3_df   = pd.read_csv('{}/ISB_O3.CSV'.format(datasets_path))
o32_df  = pd.read_csv('{}/ISB_O32.CSV'.format(datasets_path))
so2_df  = pd.read_csv('{}/ISB_SO2.CSV'.format(datasets_path))
temp_df = pd.read_csv('{}/TMP.CSV'.format(datasets_path))
humi_df = pd.read_csv('{}/RH.CSV'.format(datasets_path))

gas_range_ppb = {
    'CO':   1000000,
    'H2S':  100000,
    'NO2':  20000,
    'O3':   20000,
    'O32':  20000,
    'SO2':  100000
}

gas_obj = {
    'CO':   co_df,
    'H2S':  h2s_df,
    'NO2':  no2_df,
    'O3':   o3_df,
    'O32':  o32_df,
    'SO2':  so2_df
}

climatic_element_obj = {
    'TEMP [ºC]': temp_df,
    'RH [%]':    humi_df
}

output_dict = {}
for gas in gas_obj:
    gas_df  = gas_obj[gas]

    greatter_than_range_count = 0
    less_than_zero_count      = 0

    greatter_than_range_count = (gas_df['Value'] > gas_range_ppb[gas]).sum()
    less_than_zero_count      = (gas_df['Value'] < 0).sum()

    gas_df = gas_df[gas_df['Value'] <= gas_range_ppb[gas]]
    gas_df = gas_df[gas_df['Value'] >= 0]

    gas_var = gas_df['Value'].var()

    # Get statistic metrics and transform to dict
    statistics = gas_df['Value'].describe()
    statistics = statistics.to_dict()

    # Calc for Outliers
    lower_thr = statistics['25%'] - 1.5 * (statistics['75%'] - statistics['25%'])
    upper_thr = statistics['75%'] + 1.5 * (statistics['75%'] - statistics['25%'])

    outiler_lower_count = (gas_df['Value'] < lower_thr).sum()
    outiler_upper_count = (gas_df['Value'] > upper_thr).sum()

    # Append additional information to dict
    statistics['var']                       = gas_var
    statistics['lower_thr']                 = lower_thr
    statistics['upper_thr']                 = upper_thr
    statistics['outiler_lower_count']       = int(outiler_lower_count)
    statistics['outiler_upper_count']       = int(outiler_upper_count)
    statistics['greatter_than_range_count'] = int(greatter_than_range_count)
    statistics['less_than_zero_count']      = int(less_than_zero_count)

    output_dict[gas] = statistics

pprint(output_dict)

file_name = 'statistcs.json'
with open(file_name, 'w') as fp:
    json.dump(output_dict, fp, indent=2)