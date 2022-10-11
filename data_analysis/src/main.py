from utils        import *
from gas_analysis import *
from statistics   import *
from regressions  import *

import matplotlib.pyplot as plt
import numpy             as np

import pprint
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors 		 import KNeighborsClassifier
from sklearn.ensemble 		 import RandomForestClassifier
from scipy 					 import stats
from sklearn                 import preprocessing

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

    config      = load_config(config_path, debug_mode)
    gas_options = config["gas_sensors"]

    print("\nSelecao do gas para analise:\n")

    i = 1
    for key in gas_options:
        description = config["gas_name"][key]

        print(f"{i} - {key} ({description})")
        i += 1

    option     = get_number_by_range(1, i-1)
    gas_sensor = list(gas_options.items())[option - 1][0]
    gas_name   = config["gas_name"][gas_sensor]

    debug(f"\nGAS SELECIONADO: {gas_sensor} ({gas_name})", "green", True)

    range_ppb      = config["gas_range_ppb"][gas_sensor]
    backup_folder  = "backup_files"
    output_folder  = "output"
    df_pickle_path = "{}/{}_df_pickle".format(backup_folder, gas_sensor)

    gas_df = init_dataframe(config, gas_sensor, range_ppb, backup_folder, df_pickle_path, output_folder, debug_mode)

    # # Teste plot
    # x     = gas_df["datetime"]
    # data1 = gas_df["Value"]
    # data2 = gas_df["ma_60_gas"]
    # data3 = gas_df["ma_240_gas"]
    # data4 = gas_df["ma_720_gas"]
    # data5 = gas_df["ma_1440_gas"]

    # plot_double(x, data2, data4, title='', x_label='', data1_label='', data2_label='')

    # print(gas_df.shape)
    # print(gas_df.head())

    input_text   = "Selecao do algoritmo de analise:\n"
    options_list = ["1) Regressoes lineares", "2) K-Nearest Neighbors", "3) Random Forest", "4) Redes Neurais Artificiais", "5) Sair"]
    
    print(input_text)
    for i in options_list:
        print(i)

    option = get_number_by_range(1, len(options_list))

    if option == 1:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("---------------------------- REGRESSOES LINEARES ---------------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        lin_reg_options_list = ["1) Regressao linear simples", "2) Regressao linear multipla"]
        lin_reg_input_text   = "Tipo de regressao:\n\n" + '\n'.join(lin_reg_options_list)

        print(lin_reg_input_text)

        lin_reg_option = get_number_by_range(1, len(lin_reg_options_list))
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        if lin_reg_option == 1:
            debug("INICIALIZANDO ANALISE POR REGRESSAO LINEAR SIMPLES", debug=debug_mode)
            
            window_size = 60
            gas_df_without_nan = gas_df.dropna()
            
            # x = gas_df_without_nan[f'ma_{window_size}_temp'].tolist()
            # y = gas_df_without_nan[f'ma_{window_size}_gas'].tolist()
            # x = gas_df_without_nan['temp_value'].tolist()
            x = gas_df_without_nan['rh_value'].tolist()
            y = gas_df_without_nan['Value'].tolist()

            debug("\nSeparando dados de treinamento e teste...", "cyan", debug_mode)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            X_train = np.reshape(X_train, (-1,1))
            X_test = np.reshape(X_test, (-1,1))

            y_train = np.array(y_train)

            debug("\nCriando modelo de regressao com dados de treinamento", "cyan", debug_mode)

            regression_model = simple_linear_regression(X_train, y_train)

            debug("\nObtendo dados de teste", "cyan", debug_mode)

            y_pred = regression_model.get_predict_data(X_test)

            debug("\nObtendo metricas", "cyan", debug_mode)

            slr_dict = {}

            slr_dict["r2"]                 = regression_model.get_r2()
            b0, b1                         = regression_model.get_coefficients()
            slr_dict["coefficients"]       = {}
            slr_dict["coefficients"]["b0"] = b0
            slr_dict["coefficients"]["b1"] = b1[0]
            slr_dict["mse"]                = get_MSE(y_test, y_pred)
            slr_dict["rmse"]               = get_RMSE(y_test, y_pred)
            slr_dict["mae"]                = get_MAE(y_test, y_pred)

            debug(f'Coeficientes: {slr_dict["coefficients"]}', debug=debug_mode)
            debug(f'R^2:  {slr_dict["r2"]}', debug=debug_mode)
            debug(f'MSE:  {slr_dict["mse"]}', debug=debug_mode)
            debug(f'RMSE: {slr_dict["rmse"]}', debug=debug_mode)
            debug(f'MAE:  {slr_dict["mae"]}', debug=debug_mode)

            ouput_slr_path = "output/simple_linear_regression"

            if not os.path.exists(ouput_slr_path): os.makedirs(ouput_slr_path)

            debug(f"\nSalvando metricas no diretorio: {ouput_slr_path}", "cyan", debug_mode)

            dict_to_json_file(ouput_slr_path + "/metrics.json", slr_dict)

            debug("SUCESSO!", "green", debug_mode)

            plt.scatter(X_test, y_test, color="black")
            plt.plot(X_test, y_pred, color="blue", linewidth=3)
            plt.show()
        if lin_reg_option == 2:
            debug("INICIALIZANDO ANALISE POR REGRESSAO LINEAR MULTIPLA", debug=debug_mode)
            
            window_size = 60
            gas_df_without_nan = gas_df.dropna()

            print(gas_df_without_nan.head())

            # x = gas_df_without_nan[["temp_value", "rh_value"]]
            x = gas_df_without_nan[[f'ma_{window_size}_temp', f'ma_{window_size}_rh']]
            # y = gas_df_without_nan["Value"]
            y = gas_df_without_nan[f'ma_{window_size}_gas']
            
            debug("\nSeparando dados de treinamento e teste...", "cyan", debug_mode)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            # X_train = np.reshape(X_train, (-1,1))
            # X_test = np.reshape(X_test, (-1,1))

            y_train = np.array(y_train).reshape(-1,1)

            debug("\nCriando modelo de regressao com dados de treinamento", "cyan", debug_mode)

            regression_model = simple_linear_regression(X_train, y_train)

            debug("\nObtendo dados de teste", "cyan", debug_mode)

            y_pred = regression_model.get_predict_data(X_test)

            debug("\nObtendo metricas", "cyan", debug_mode)

            mlr_dict = {}

            mlr_dict["r2"]           = regression_model.get_r2()
            b0, b1                   = regression_model.get_coefficients()
            b1                       = b1[0].tolist()
            mlr_dict["coefficients"] = {}
            mlr_dict["coefficients"]["b0"] = b0[0]
            mlr_dict["coefficients"]["b1"] = b1
            mlr_dict["mse"]          = get_MSE(y_test, y_pred)
            mlr_dict["rmse"]         = get_RMSE(y_test, y_pred)
            mlr_dict["mae"]          = get_MAE(y_test, y_pred)

            debug(f'Coeficientes: {mlr_dict["coefficients"]}', debug=debug_mode)
            debug(f'R^2:  {mlr_dict["r2"]}', debug=debug_mode)
            debug(f'MSE:  {mlr_dict["mse"]}', debug=debug_mode)
            debug(f'RMSE: {mlr_dict["rmse"]}', debug=debug_mode)
            debug(f'MAE:  {mlr_dict["mae"]}', debug=debug_mode)

            ouput_slr_path = "output/multiple_linear_regression"

            if not os.path.exists(ouput_slr_path): os.makedirs(ouput_slr_path)

            debug(f"\nSalvando metricas no diretorio: {ouput_slr_path}", "cyan", debug_mode)

            dict_to_json_file(ouput_slr_path + "/metrics.json", mlr_dict)

            debug("SUCESSO!", "green", debug_mode)

            plt.scatter(X_test, y_test, color="black")
            plt.plot(X_test, y_pred, color="blue", linewidth=3)
            plt.show()
    if option == 2:
        gas_df_without_nan = gas_df.dropna()

        print(gas_df_without_nan.head())

        x = gas_df_without_nan[["temp_value", "rh_value"]].values
        # y = gas_df_without_nan["Value"].values
        # x = gas_df_without_nan[["ma_60_temp", "ma_60_rh"]].values
        y = gas_df_without_nan["ma_60_gas"].values

        max_neighbors = 25
        neighbors     = np.arange(1, max_neighbors)

        test_accuracy_knn  = np.empty(len(neighbors))
        knn_precision_list = []
        rf_precision_list  = []
        k_variation 	   = []

        i = 0
        while i < 2:
            print(f"INCIANDO LACO {i}\n\n")

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            y_train = np.ravel(y_train)
            y_test  = np.ravel(y_test)
            
            lab_enc = preprocessing.LabelEncoder()
            y_train = lab_enc.fit_transform(y_train)
            y_test  = lab_enc.fit_transform(y_test)

            for j,k in enumerate(neighbors):
                knn = KNeighborsClassifier(n_neighbors=k)
            
                #divide o modelo
                knn.fit(X_train, y_train)	
                    
                # Calcula a precisao dos dados de teste
                test_accuracy_knn[j] = knn.score(X_test, y_test)

            knn_precision_list.append(max(test_accuracy_knn))
            k_variation.append(np.argmax(test_accuracy_knn))

            i += 1

        k_variation = np.array(k_variation)
        media_knn 	= np.mean(k_variation)
        std_knn 	= np.std(k_variation)

        knn_precision_list = np.array(knn_precision_list)

        knn_media = np.mean(knn_precision_list)

        print("-------K-NN-------")
        print("Media: " + str(knn_media))

        print("-------Vizinhos-------")
        print("Media: " + str(media_knn))
        print("Min: " + str(media_knn - std_knn))
        print("Max: " + str(media_knn + std_knn))
        print("numd e vizinhos mais repetido: " + str(stats.mode(k_variation, keepdims = True)))

        # Verificando precisao do K-NN e Random Forest para diferentes valores de K e arvores de decisao
        fig, ax1 = plt.subplots()

        ax1.set_title('k-NN variando numero de vizinhos proximos', size=25)
        ax1.plot(neighbors, test_accuracy_knn, label='Precisao K-NN')
        ax1.set_xlabel('Numero de vizinhos proximos', size=25)
        ax1.set_ylabel('Acuracia', size=25)
        plt.grid()
        plt.show()

        fig.tight_layout()

        ouput_knn_path = "output/knn"

        if not os.path.exists(output_folder): os.makedirs(output_folder)
        if not os.path.exists(ouput_knn_path): os.makedirs(ouput_knn_path)

        fig.savefig('{}/{}_{}.png'.format(ouput_knn_path, "knn", "variation"), dpi=fig.dpi)

    if option == 3:
        gas_df_without_nan = gas_df.dropna()

        print(gas_df_without_nan.head())

        x = gas_df_without_nan[["temp_value", "rh_value"]].values
        # y = gas_df_without_nan["Value"].values
        # x = gas_df_without_nan[["ma_60_temp", "ma_60_rh"]].values
        y = gas_df_without_nan["ma_60_gas"].values

        max_trees = 25
        trees 	  = np.arange(1,max_trees)

        test_accuracy_random_forest  = np.empty(len(trees))
        rf_precision_list            = []
        trees_variation              = []

        i = 0
        while i < 2:
            print(f"INCIANDO LACO {i}\n\n")

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            y_train = np.ravel(y_train)
            y_test  = np.ravel(y_test)

            lab_enc = preprocessing.LabelEncoder()
            y_train = lab_enc.fit_transform(y_train)
            y_test  = lab_enc.fit_transform(y_test)

            for j,k in enumerate(trees):
                random_forest = RandomForestClassifier(n_estimators=k)  

                #divide o modelo
                random_forest.fit(X_train, y_train)

                # Calcula a precisao dos dados de teste
                test_accuracy_random_forest[j] = random_forest.score(X_test, y_test)

            rf_precision_list.append(max(test_accuracy_random_forest))
            trees_variation.append(np.argmax(test_accuracy_random_forest))

            i += 1

        trees_variation = np.array(trees_variation)
        media_rf 		= np.mean(trees_variation)
        std_rf 			= np.std(trees_variation)

        rf_precision_list  = np.array(rf_precision_list)
        rf_media           = np.mean(rf_precision_list)

        print("-------Random Forest-------")
        print("Media: " + str(rf_media))

        print("-------Arvores-------")
        print("Media: " + str(media_rf))
        print("Min: " + str(media_rf - std_rf))
        print("Max: " + str(media_rf + std_rf))

        print("num de arvores mais repetido: " + str(stats.mode(trees_variation, keepdims = True)))

       # Verificando acuracia do Random Forest para diferentes valores de arvores de decisao
        fig, ax1 = plt.subplots()

        ax1.set_title('Random Forest variando numero de vizinhos proximos', size=25)
        ax1.plot(trees, test_accuracy_random_forest, label='Acuracia Random Forest')
        ax1.set_xlabel('Numero de arvores de decisao', size=25)
        ax1.set_ylabel('Acuracia', size=25)
        plt.grid()
        plt.show()

        fig.tight_layout()

        output_rf_path = "output/random_forest"

        if not os.path.exists(output_folder): os.makedirs(output_folder)
        if not os.path.exists(output_rf_path): os.makedirs(output_rf_path)

        fig.savefig('{}/{}_{}.png'.format(output_rf_path, "trees", "variation"), dpi=fig.dpi)

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
    while True:
        main()