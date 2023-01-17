from utils import *
import pandas as pd
import sys

class Gas_Analysis:
    def __init__(self, gas_df, config, gas_sensor, debug_mode):
        self.gas_df       = gas_df
        self.gas_analysis = gas_sensor

        features_data             = config["features_data"]
        datasets_features_names   = list(features_data.keys())
        datasets_features_paths   = [features_data[k][0] for k in datasets_features_names]
        datasets_features_columns = [features_data[k][1] for k in datasets_features_names]

        debug("Starting features dataframes...", "blue", debug_mode)

        try:
            df_features = []

            for i in datasets_features_paths:
                df_features.append(pd.read_csv(i))

            debug("Success!", "green", debug_mode)
        except Exception as e:
            debug("\nFalha ao carregar datasets!", "red", debug_mode)
            print(e)

            sys.exit(0)

        debug("\nAdding features data to the pollutant dataset...", "blue", debug_mode)

        i = 0
        while i < len(datasets_features_names):
            self.append_data_from_df(df_features[i], datasets_features_columns[i], datasets_features_names[i])

            i += 1

        # debug("Dataframe do poluente {} atualizado com sucesso!".format(config["gas_name"][self.gas_analysis]), "green", debug_mode)

    def remove_unused_columns(self, unused_columns_list):
        self.gas_df = self.gas_df.drop(unused_columns_list, axis=1)

    def append_data_from_df(self, src_df, src_df_column, new_column_id):
        gas_df_length = self.gas_df.shape[0]
        src_df_length = src_df.shape[0]
        len_diff      = abs(gas_df_length - src_df_length)

        if gas_df_length < src_df_length:
            src_df.drop(src_df.tail(len_diff).index, inplace = True)
        elif gas_df_length > src_df_length:
            self.gas_df.drop(self.gas_df.tail(len_diff).index, inplace = True)

        self.gas_df[new_column_id] = src_df[src_df_column]