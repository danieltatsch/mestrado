######################################################
# Plot duplo - media movel
######################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

import datetime

climatic_element = 'TEMP [ºC]'
# climatic_element   = 'RH [%]'
gas_sensor         = 'CO'

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

gas_df              = gas_obj[gas_sensor]
climatic_element_df = climatic_element_obj[climatic_element]

gas_df_length           = gas_df.shape[0]
climatic_element_length = climatic_element_df.shape[0]

# TODO: Remover outliers

date_time_col     = ['Year','Month','Day','Hour','Minute','Second']
gas_df['DateTime'] = (pd.to_datetime(gas_df[date_time_col],
                                    infer_datetime_format=False,
                                    format='%d/%m/%Y/%H/%M/%S'))

gas_df_datatime = gas_df['DateTime'].values

gas_values = gas_df['Value']

gas_values_np = np.array(gas_values.values.tolist())

# Replace values > sensor range by max value
gas_values    = np.where(gas_values_np > gas_range_ppb[gas_sensor], gas_range_ppb[gas_sensor], gas_values_np)

# Replace values < 0 by 0
gas_values    = np.where(gas_values_np < 0, 0, gas_values_np).tolist()

gas_df        = pd.DataFrame({'DateTime': gas_df_datatime, 'Value': gas_values})

ma_window = 1
climatic_element_df['moving_average'] = climatic_element_df['Value'].rolling(ma_window).mean()
gas_df['moving_average']              = gas_df['Value'].rolling(ma_window).mean()

x     = gas_df['DateTime']
data1 = gas_df['moving_average']
data2 = climatic_element_df['moving_average']

if gas_df_length < climatic_element_length:
    data2 = climatic_element_df['moving_average'][:gas_df_length]
    x     = gas_df['DateTime'][:gas_df_length]
elif gas_df_length > climatic_element_length:
    data1 = gas_df['moving_average'][:climatic_element_length]
    x     = gas_df['DateTime'][:climatic_element_length]

fig, ax1 = plt.subplots()
ax1.set_title('Média móvel: {} [ppb] x {}'.format(gas_sensor, climatic_element), size=25)
color = 'tab:blue'
ax1.set_xlabel('tempo [min]', size=25)
ax1.set_ylabel('{} - concentração [ppb]'.format(gas_sensor), color=color, size=25)
ax1.plot(x, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
ax1.tick_params(axis='x', labelsize=15)
plt.grid(color='blue', which='both')
plt.xticks(rotation=45)

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel('{}'.format(climatic_element), color=color, size=25)
ax2.plot(x, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=20)
ax2.tick_params(axis='x', labelsize=15)
plt.grid(color='orange', which='both')

plt.xticks(rotation=45)

fig.tight_layout()
plt.show()