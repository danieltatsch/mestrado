from utils        import *
from gas_analysis import *
from statistics   import *

import pprint
import sys
import os

debug_mode  = True
config_path = "config.json"

def init_dataframe(config, gas_sensor, range_ppb, backup_folder, df_pickle_path, output_folder, debug_mode):
    gas_df = None

    if os.path.exists(df_pickle_path):
        debug("\nInicializando dataframe a partir do backup: {}".format(df_pickle_path), "cyan", debug_mode)

        gas_df = pd.read_pickle(df_pickle_path)

        debug("Sucesso!", "green", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        return gas_df
    else:
        gas_analysis = Gas_Analysis(config, gas_sensor, debug_mode)
        gas_analysis.remove_unused_columns()

        gas_df         = gas_analysis.gas_df.copy()
        ma_window_list = config["moving_average_window"]

        debug("\nRealizando calculo da media movel com as janelas: {}".format(ma_window_list), "cyan", debug_mode)

        for window in ma_window_list:
            gas_df["ma_{}_gas".format(window)]  = gas_df["Value"].rolling(window).mean()
            gas_df["ma_{}_temp".format(window)] = gas_df["temp_value"].rolling(window).mean()
            gas_df["ma_{}_rh".format(window)]   = gas_df["rh_value"].rolling(window).mean()

        print(gas_df.head())

        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("---------------------- REALIZANDO ANALISE ESTATISTICA ----------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)
        
        gas_df_processed = remove_invalid_data(gas_df, range_ppb, gas_sensor, output_folder, debug_mode)

        debug(f"Salvando dataset no arquivo de backup {df_pickle_path}", "cyan", debug_mode)

        if not os.path.exists(backup_folder): os.makedirs(backup_folder)
        gas_df.to_pickle(df_pickle_path)

        debug("Sucesso!", "green", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        return gas_df_processed

def main():
    debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
    debug("----------------------- INICIALIZACAO DO EXPERIMENTO -----------------------", "cyan", debug_mode)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    config         = load_config(config_path, debug_mode)
    gas_sensor     = config["default_gas_df"]
    range_ppb      = config["gas_range_ppb"][gas_sensor]
    backup_folder  = "backup_files"
    output_folder  = "output"
    df_pickle_path = "{}/{}_df_pickle".format(backup_folder, gas_sensor)

    gas_df = init_dataframe(config, gas_sensor, range_ppb, backup_folder, df_pickle_path, output_folder, debug_mode)

    # print(gas_df.shape)
    # print(gas_df.head())

    options_list = ["1) Regressoes lineares", "2) K-Nearest Neighbors", "3) Random Forest", "4) Redes Neurais Artificiais", "5) Sair"]
    input_text   = "Selecione a opcao desejada:\n" + '\n'.join(options_list)

    option = get_number_by_range(1, len(options_list), input_text)

    if option == 1:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("---------------------------- REGRESSOES LINEARES ---------------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    if option == 5:
        sys.exit()

def remove_invalid_data(gas_df, range_ppb, gas_sensor, output_folder, debug_mode):
    gas_value  = gas_df["Value"]
    temp_value = gas_df["temp_value"]
    rh_value   = gas_df["rh_value"]

    debug("Dados sem processamento...", "cyan", debug_mode)
    statistics_raw = get_statistics_data(gas_value, temp_value, rh_value, range_ppb)
    pprint.pprint(statistics_raw)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    debug("Dados processados...", "cyan", debug_mode)

    debug("\nRemovendo dados fora dos limiares do sensor {}".format(gas_sensor), "blue", debug_mode)
    gas_df_processed = remove_values_outside_range(gas_df, gas_value, range_ppb)

    gas_value_proc  = gas_df_processed["Value"]
    temp_value_proc = gas_df_processed["temp_value"]
    rh_value_proc   = gas_df_processed["rh_value"]

    statistics_proc = get_statistics_data(gas_value_proc, temp_value_proc, rh_value_proc, range_ppb)

    debug("\nRemovendo outliers".format(gas_sensor), "blue", debug_mode)
    gas_df_processed_out = remove_outliers_from_df(gas_df_processed, gas_value_proc, statistics_proc["lower_thr"], statistics_proc["upper_thr"])

    gas_value_out  = gas_df_processed_out["Value"]
    temp_value_out = gas_df_processed_out["temp_value"]
    rh_value_out   = gas_df_processed_out["rh_value"]

    statistics_processed = get_statistics_data(gas_value_out, temp_value_out, rh_value_out, range_ppb)

    pprint.pprint(statistics_processed)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    output_file_raw       = "{}/statistics_raw_{}".format(output_folder, gas_sensor)
    output_file_processed = "{}/statistics_processed_{}".format(output_folder, gas_sensor)

    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    dict_to_json_file(output_file_raw, statistics_raw)
    dict_to_json_file(output_file_processed, statistics_processed)

    return gas_df_processed_out

def get_statistics_data(gas_value, temp_value, rh_value, range_ppb):
    debug("\nContando dados fora dos limiares", "blue", debug_mode)
    greatter_than_range_count, less_than_zero_count = count_values_outside_range(gas_value, range_ppb)

    debug("\nCalculando metricas", "blue", debug_mode)
    statistics = get_statistics_by_describe(gas_value)

    debug("\nCalculando correlacao com TEMP e RH", "blue", debug_mode)
    corr_gas_temp = get_correlation(gas_value, temp_value)
    corr_gas_rh   = get_correlation(gas_value, rh_value)

    debug("\nCalculando outliers", "blue", debug_mode)
    outliers_info = get_outliers(gas_value, statistics["25%"], statistics["75%"])

    debug("\nSucesso! Registrando informacoes...\n", "green", debug_mode)
    statistics['greatter_than_range_count'] = int(greatter_than_range_count)
    statistics['less_than_zero_count']      = int(less_than_zero_count)
    statistics['corr_gas_temp']             = corr_gas_temp
    statistics['corr_gas_rh']               = corr_gas_rh
    statistics = {** statistics, **outliers_info}

    return statistics

if __name__ == "__main__":
    main()