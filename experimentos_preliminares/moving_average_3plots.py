import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np 

fig = plt.figure()
host = fig.add_subplot(111)

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

climatic_element_df['moving_average'] = climatic_element_df['Value'].rolling(window=1440).mean()
humi_df['moving_average']             = humi_df['Value'].rolling(window=1440).mean()
gas_df['moving_average']              = gas_df['Value'].rolling(window=1440).mean()

data1 = gas_df['moving_average']
data2 = climatic_element_df['moving_average']
data3 = humi_df['moving_average']

par1 = host.twinx()
par2 = host.twinx()
    
host.set_xlabel("Tempo [min]")
host.set_ylabel("CO - concentracao [ppb]")
par1.set_ylabel("TEMP [ºC]")
par2.set_ylabel("RH [%]")

color1 = 'blue'
color2 = 'red'
color3 = 'orange'

p1, = host.plot(data1, color=color1, label="CO")
p2, = par1.plot(data2, color=color2, label="TEMP")
p3, = par2.plot(data3, color=color3, label="RH")

lns = [p1, p2, p3]
host.legend(handles=lns, loc='best')

par2.spines['right'].set_position(('outward', 60))

par2.xaxis.set_ticks([])

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

fig.tight_layout()
plt.show()