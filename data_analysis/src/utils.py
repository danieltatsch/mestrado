import json
import colorama
from   termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd

def open_json_file(path : str) -> dict:
    data = {}

    try:
        with open(path) as json_file:
            data = json.load(json_file)
    except:
        pass

    return data

def dict_to_json_file(out_file_name, out_dict):
    with open(out_file_name, 'w') as fp:
        json.dump(out_dict, fp, indent=2)

def load_config(config_path, debug_mode):
    debug("Abrindo arquivo de configuracao...", "blue", debug_mode)
    config = open_json_file(config_path)

    if not config:
        debug("\nFalha ao abrir arquivo\n", "red", debug_mode)

    return config

def debug(text, color=None, debug=True):
    if debug:
        print(colored(text, color)) if color is not None else print(text)

def confirm_info(text=None):
    input_options_list = ["s", "1", "n", "0"]
    input_text         = "\nConfirmar informacoes? (S,1/N,0): " if text is None else f"\n{text} (S,1/N,0): "

    while True:
        option_str = input(input_text)

        if option_str in input_options_list:
            return option_str == "s" or option_str == "1"

        debug("\nEntrada invalida!", "red", True)

def get_number_by_range(lower_th, upper_th, text=None):
    input_text = text if text is not None else f"Selecione um numero entre {lower_th} e {upper_th}: "

    option = -1
    while True:
        try:
            # print(input_text)
            option = int(input(f"\n{input_text}"))
        except:
            pass

        if option >= lower_th and option <= upper_th:
            break

        debug("\nEntrada invalida!", "red", True)

    return option

def set_datatime_col(gas_df):
    date_time_df             = pd.DataFrame()
    date_time_col            = ['Year','Month','Day','Hour','Minute','Second']
    date_time_df['datetime'] = (pd.to_datetime(gas_df[date_time_col], infer_datetime_format=False, format='%d/%m/%Y/%H/%M/%S'))

    return date_time_df

def plot_double(x, data1, data2, title='', x_label='', data1_label='', data2_label=''):
    # x     = gas_df_ma["datetime"]
    # data1 = gas_df_ma["ma_gas"]
    # data2 = gas_df_ma["temp_value"]
    # data3 = gas_df_ma["ma_rh"]

    # climatic_element = "TEMP"
    # # climatic_element = "RH"
    # title       = 'Média móvel: {} [ppb] x {}'.format(gas_sensor, climatic_element)
    # x_label     = 'Data'
    # data1_label = '{} - concentração [ppb]'.format(gas_sensor)
    # data2_label = 'RH [%]' if climatic_element == 'RH' else 'TEMP [ºC]'

    # plot_double(x, data1, data2, title, x_label, data1_label, data2_label)

    fig, ax1 = plt.subplots()
    ax1.set_title(title, size=25)
    color = 'tab:blue'
    ax1.set_xlabel(x_label, size=25)
    ax1.set_ylabel(data1_label, color=color, size=25)
    ax1.plot(x, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    plt.grid(color='blue', which='both')
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()

    color = 'tab:orange'
    ax2.set_ylabel(data2_label, color=color, size=25)
    ax2.plot(x, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    plt.grid(color='orange', which='both')

    plt.xticks(rotation=45)

    fig.tight_layout()
    plt.show()