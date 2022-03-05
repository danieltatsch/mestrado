######################################################
# Plot duplo - media movel
######################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

climatic_element = 'TEMP [ºC]'
# climatic_element   = 'RH [%]'
gas_sensor         = 'CO'

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

gas_df              = gas_obj[gas_sensor]
climatic_element_df = climatic_element_obj[climatic_element]

# TODO: Substituir por remocao dos outliers
# gas_mean   = gas_df['Value'].mean()
# gas_values = gas_df['Value']

# gas_values_np = np.array(gas_values.values.tolist())
# gas_values    = np.where(gas_values_np > 25000, gas_mean, gas_values_np).tolist()
# gas_df        = pd.DataFrame({'Value': gas_values})

climatic_element_df['moving_average'] = climatic_element_df['Value'].rolling(window=1440).mean()
gas_df['moving_average']              = gas_df['Value'].rolling(window=1440).mean()

data1 = gas_df['moving_average']
data2 = climatic_element_df['moving_average']

fig, ax1 = plt.subplots()
ax1.set_title('Média móvel: {} [ppb] x {}'.format(gas_sensor, climatic_element), size=25)
color = 'tab:blue'
ax1.set_xlabel('tempo [min]', size=25)
ax1.set_ylabel('{} - concentração [ppb]'.format(gas_sensor), color=color, size=25)
ax1.plot(data1, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
ax1.tick_params(axis='x', labelsize=20)

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel('{}'.format(climatic_element), color=color, size=25)
ax2.plot(data2, color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=20)
ax2.tick_params(axis='x', labelsize=20)

fig.tight_layout()
plt.show()