import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from regressions             import simple_linear_regression

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

gas_df  = gas_obj['O3']

# ma_window = 240
# gas_df['moving_average'] = gas_df['Value'].rolling(ma_window).mean()
# values  = gas_df['moving_average'].dropna()
values  = gas_df['Value'].to_list()

print(gas_df.describe)

input()

indexes = list(range(len(values)))

window_size = 240
values_by_window  = [values[i:i+window_size] for i in range(0, len(values), window_size)]  
indexes_by_window = [indexes[i:i+window_size] for i in range(0, len(indexes), window_size)]  

# values  = values[:window_size]
# indexes = indexes[:window_size]
for i in range(len(values_by_window)):
    X_train, X_test, y_train, y_test = train_test_split(indexes_by_window[i], values_by_window[i], test_size=0.33, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = np.reshape(X_train, (-1,1))
    X_test = np.reshape(X_test, (-1,1))

    y_train = np.array(y_train)

    regression_model = simple_linear_regression(X_train, y_train)

    b0, b1 = regression_model.get_coefficients()

    print(regression_model.get_coefficients())

    y_pred = regression_model.get_predict_data(X_test)

    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=3)
# plt.xticks(())
# plt.yticks(())

plt.show()