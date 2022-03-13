######################################################
# Plot duplo - dados temporais
######################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

datasets_path = 'dados_lcqar_07_08_2020'

co_df   = pd.read_csv('{}/CO.CSV'.format(datasets_path))
no2_df  = pd.read_csv('{}/NO2.CSV'.format(datasets_path))
o3_df   = pd.read_csv('{}/O3.CSV'.format(datasets_path))
so2_df  = pd.read_csv('{}/SO2.CSV'.format(datasets_path))
temp_df = pd.read_csv('{}/TMP.CSV'.format(datasets_path))
humi_df = pd.read_csv('{}/RH.CSV'.format(datasets_path))

mean_co_df = co_df['Value'].mean()
values_co_df = co_df['Value']
# co_df['Value'] = co_df['Value'].apply(lambda x: [y if y <= mean_co_df else mean_co_df for y in x])

a            = np.array(values_co_df.values.tolist())
values_co_df = np.where(a > 25000, mean_co_df, a).tolist()

data1 = values_co_df
data2 = humi_df['Value']

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('tempo [min]')
ax1.set_ylabel('concentracao [ppb]', color=color)
ax1.plot(data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('RH [%]', color=color)  # we already handled the x-label with ax1
ax2.plot(data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()