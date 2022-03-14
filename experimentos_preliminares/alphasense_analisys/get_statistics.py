import pandas  as pd
import numpy   as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats  import pearsonr
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
    temp_df = climatic_element_obj['TEMP [ºC]']
    rh_df   = climatic_element_obj['RH [%]']

    gas_df_length  = gas_df.shape[0]
    rh_df_length   = rh_df.shape[0]
    temp_df_length = temp_df.shape[0]

    # Set datasets to the same size
    if gas_df_length < temp_df_length:
        temp_df = temp_df[:gas_df_length]
        rh_df   = rh_df[:gas_df_length]
    elif gas_df_length > temp_df_length:
        gas_df = gas_df[:temp_df_length]

    print('GAS DF SIZE:  {}'.format(gas_df.shape[0]))
    print("TEMP DF SIZE: {}".format(temp_df.shape[0]))
    print("RH DF SIZE:   {}\n".format(rh_df.shape[0]))

    print("-------------------------------")

    # Count number of values outside the sensor range
    greatter_than_range_count = (gas_df['Value'] > gas_range_ppb[gas]).sum()
    less_than_zero_count      = (gas_df['Value'] < 0).sum()

    # Get outside values range to exclude the correct indexes of RH and TEMP datasets
    greatter_than_range_index = gas_df[gas_df['Value'] > gas_range_ppb[gas]].index.tolist()
    less_than_range_index     = gas_df[gas_df['Value'] < 0].index.tolist()

    # Remove indexes from RH and TEMP datasets
    temp_df = temp_df.drop(temp_df.index[greatter_than_range_index])
    rh_df   = rh_df.drop(rh_df.index[greatter_than_range_index])

    temp_df = temp_df.drop(temp_df.index[less_than_range_index])
    rh_df   = rh_df.drop(rh_df.index[less_than_range_index])

    # Remove values outside the sensor range
    gas_df = gas_df[gas_df['Value'] <= gas_range_ppb[gas]]
    gas_df = gas_df[gas_df['Value'] >= 0]

    # Correlation with outliers
    corr_temp_w_outliers, _ = pearsonr(gas_df['Value'], temp_df['Value'])
    corr_rh_w_outliers, _   = pearsonr(gas_df['Value'], rh_df['Value'])
    # print('{} CORR TEMP: %.3f'.format(gas) % corr_temp_w_outliers)
    # print('{} CORR RH:   %.3f\n'.format(gas) % corr_rh_w_outliers)

    # Get variance
    gas_var = gas_df['Value'].var()

    # Get statistic metrics and transform to dict
    statistics = gas_df['Value'].describe()
    statistics = statistics.to_dict()

    # Calc for Outliers
    lower_thr = statistics['25%'] - 1.5 * (statistics['75%'] - statistics['25%'])
    upper_thr = statistics['75%'] + 1.5 * (statistics['75%'] - statistics['25%'])

    outiler_lower_count = (gas_df['Value'] < lower_thr).sum()
    outiler_upper_count = (gas_df['Value'] > upper_thr).sum()

    # Remove outliers from gas_df, temp_df and rh_df
    gas_df_without_outliers = gas_df[gas_df['Value'] <= upper_thr]
    gas_df_without_outliers = gas_df_without_outliers[gas_df_without_outliers['Value'] >= lower_thr]

    outliers_indexes_list = gas_df[gas_df['Value'] >= upper_thr].index.tolist()
    outliers_indexes_list = outliers_indexes_list + gas_df_without_outliers[gas_df_without_outliers['Value'] <= lower_thr].index.tolist()

    # Correlation without outliers
    temp_df = temp_df.drop(temp_df.index[outliers_indexes_list])
    rh_df   = rh_df.drop(rh_df.index[outliers_indexes_list])

    corr_temp_wo_outliers, _ = pearsonr(gas_df_without_outliers['Value'], temp_df['Value'])
    corr_rh_wo_outliers, _   = pearsonr(gas_df_without_outliers['Value'], rh_df['Value'])
    print('{} CORR TEMP: %.3f'.format(gas) % corr_temp_wo_outliers)
    print('{} CORR RH:   %.3f\n'.format(gas) % corr_rh_wo_outliers)

    input("FIM DA CORRELACAO SEM OS OUTLIERS")

    # Append additional information to dict
    statistics['var']                       = gas_var
    statistics['lower_thr']                 = lower_thr
    statistics['upper_thr']                 = upper_thr
    statistics['outiler_lower_count']       = int(outiler_lower_count)
    statistics['outiler_upper_count']       = int(outiler_upper_count)
    statistics['greatter_than_range_count'] = int(greatter_than_range_count)
    statistics['less_than_zero_count']      = int(less_than_zero_count)
    statistics['temp_cor_outliers']         = corr_temp_w_outliers
    statistics['rh_cor_outlier']            = corr_rh_w_outliers

    # boxplot without outliers

    # sns.set_theme(style="whitegrid")
    # ax = sns.boxplot(x=gas_df['Value'], showfliers = False)
    # ax = sns.boxplot(x=gas_df['Value'])
    # ax.axes.set_title('Boxplot - {}'.format(gas),fontsize=25)
    # ax.set_xlabel('Concentração [ppb]',fontsize=25)
    # ax.tick_params(axis='x', labelsize=15)
    # plt.show()

    # histogram

    # sns.set_theme(style="whitegrid")
    # ax = sns.histplot(x=gas_df_without_outliers['Value'])
    # # ax = sns.histplot(x=gas_df['Value'])
    # ax.axes.set_title('Histograma - {}'.format(gas),fontsize=25)
    # ax.set_xlabel('Concentração [ppb]',fontsize=25)
    # ax.set_ylabel('Quantidade de medições',fontsize=25)
    # ax.tick_params(axis='x', labelsize=15)
    # plt.show()

    # correlation

    # statistic analisys

    output_dict[gas] = statistics
    # gas_obj[gas]     = gas_df

pprint(output_dict)

file_name = 'statistcs.json'
with open(file_name, 'w') as fp:
    json.dump(output_dict, fp, indent=2)