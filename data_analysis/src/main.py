from utils        import *
from gas_analysis import *
import os

debug_mode  = True
config_path = "config.json"

def init_dataframe(config, gas_sensor, backup_folder, df_pickle_path, debug_mode):
    gas_df = None

    if os.path.exists(df_pickle_path):
        debug("\nInicializando dataframe a partir do backup: {}".format(df_pickle_path), "blue", debug_mode)

        gas_df = pd.read_pickle(df_pickle_path)

        debug("Sucesso!", "green", debug_mode)
    else:
        gas_analysis = Gas_Analysis(config, gas_sensor, debug_mode)
        gas_analysis.remove_unused_columns()

        gas_df         = gas_analysis.gas_df.copy()
        ma_window_list = config["moving_average_window"]

        debug("\nRealizando calculo da media movel com as janelas: {}".format(ma_window_list), "blue", debug_mode)

        for window in ma_window_list:
            gas_df["ma_{}_gas".format(window)]  = gas_df["Value"].rolling(window).mean()
            gas_df["ma_{}_temp".format(window)] = gas_df["temp_value"].rolling(window).mean()
            gas_df["ma_{}_rh".format(window)]   = gas_df["rh_value"].rolling(window).mean()

        print(gas_df.head())

        debug("Salvando dataset com as medias moveis", "blue", debug_mode)

        if not os.path.exists(backup_folder): os.makedirs(backup_folder)
        gas_df.to_pickle(df_pickle_path)

        debug("Sucesso!", "green", debug_mode)

    return gas_df

def main():
    debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
    debug("----------------------- INICIALIZACAO DO EXPERIMENTO -----------------------", "cyan", debug_mode)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    config         = load_config(config_path, debug_mode)
    gas_sensor     = config["default_gas_df"]
    backup_folder  = "backup_files"
    df_pickle_path = "{}/{}_df_pickle".format(backup_folder, gas_sensor)

    gas_df = init_dataframe(config, gas_sensor, backup_folder, df_pickle_path, debug_mode)

if __name__ == "__main__":
    main()