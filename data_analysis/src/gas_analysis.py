from utils import *
import pandas as pd
import sys

config_path = "config.json"

class Gas_Analysis:
    def __init__(self, debug_mode):
        self.config = None
        self.gas_df = None

        self.load_config(config_path, debug_mode)

        self.gas_analysis = self.config["default_gas_df"]

        gas_df_path  = self.config["gas_sensors"][self.gas_analysis]
        temp_df_path = self.config["climatic_elements"]["TEMP"]
        rh_df_path   = self.config["climatic_elements"]["RH"]

        debug("Inicializando dataframes...\n", "blue", debug_mode)
        debug("Gas em analise: {}".format(gas_df_path), None, debug_mode)
        debug("Dataset temp:   {}".format(temp_df_path), None, debug_mode)
        debug("Dataset RH:     {}".format(rh_df_path), None, debug_mode)

        try:
            self.gas_df = pd.read_csv(gas_df_path)
            temp_df     = pd.read_csv(temp_df_path)
            rh_df       = pd.read_csv(rh_df_path)

            debug("\nSucesso!", "green", debug_mode)
        except Exception as e:
            debug("\nFalha ao carregar datasets!", "red", debug_mode)
            print(e)

            sys.exit(0)

        debug("\nInserindo campos de datetime, temperatura e umidade ao dataset em analise...\n", "blue", debug_mode)
        self.set_datatime_col()
        self.append_data_from_df(temp_df, "Value", "temp_value")
        self.append_data_from_df(rh_df, "Value", "rh_value")
        debug("Dataframe do poluente {} inicializado com sucesso!".format(self.config["gas_name"][self.gas_analysis]), "green", debug_mode)

    def remove_unused_columns(self):
        self.gas_df = self.gas_df.drop(["Year", "Month", "Day", "Hour", "Minute", "Second", "Latitude", "Longitude", "Altitude", "Device", "DeviceSt"], axis=1)

    def append_data_from_df(self, src_df, src_df_column, new_column_id):
        gas_df_length = self.gas_df.shape[0]
        src_df_length = src_df.shape[0]
        len_diff      = abs(gas_df_length - src_df_length)

        if gas_df_length < src_df_length:
            src_df.drop(src_df.tail(len_diff).index, inplace = True)
        elif gas_df_length > src_df_length:
            self.gas_df.drop(self.gas_df.tail(len_diff).index, inplace = True)

        self.gas_df[new_column_id] = src_df[src_df_column]

    def load_config(self, config_path, debug_mode):
        debug("Abrindo arquivo de configuracao...", "blue", debug_mode)
        self.config = open_json_file(config_path)

        if not self.config:
            debug("\nFalha ao abrir arquivo\n", "red", debug_mode)

    def set_datatime_col(self):
        date_time_col           = ['Year','Month','Day','Hour','Minute','Second']
        self.gas_df['datetime'] = (pd.to_datetime(self.gas_df[date_time_col], infer_datetime_format=False, format='%d/%m/%Y/%H/%M/%S'))