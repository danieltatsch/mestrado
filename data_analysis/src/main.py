from utils        import *
from gas_analysis import *
from statistics   import *
from regressions  import *

import matplotlib.pyplot as plt
import numpy             as np

import math
import pprint
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors 		 import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble 		 import RandomForestClassifier
from sklearn.metrics 		 import confusion_matrix, classification_report
from sklearn.svm 			 import SVC, SVR, LinearSVR

from tensorflow.keras.models    import Sequential, load_model
from tensorflow.keras.layers    import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics   import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError, MeanAbsoluteError
from tensorflow.keras.utils     import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics 		 import confusion_matrix, classification_report
from sklearn.utils           import class_weight

debug_mode  = True
config_path = "config.json"

def load_dataframe(config, gas_sensor, df_pickle_path, bkp_available):
    gas_df = None

    # if bkp_available:
    #     debug("\nInicializando dataframe a partir do backup: {}".format(df_pickle_path), "cyan", debug_mode)

    #     gas_df = pd.read_pickle(df_pickle_path)

    #     debug("Sucesso!", "green", debug_mode)
    #     debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)
    # else:
    gas_df_path  = config["gas_sensors"][gas_sensor]

    debug(f"Inicializando dataframe a partir do arquivo {gas_df_path}", "blue", debug_mode)

    try:
        gas_df = pd.read_csv(gas_df_path)

        debug("Sucesso!\n", "green", debug_mode)
    except Exception as e:
        debug("Falha ao carregar dataset!\n", "red", debug_mode)
        print(e)

        sys.exit(0)

    return gas_df

def apply_moving_average(gas_df_raw, ma_window, target_value, features_list):
    gas_df          = pd.DataFrame()
    df_columns_list = features_list.copy()
    df_columns_list.append(target_value)

    debug(f"\nRealizando calculo da media movel com a janela de {ma_window} pontos", "cyan", debug_mode)

    new_columns_list = []

    for i in df_columns_list:
        new_columns_list.append(f"ma_{ma_window}_{i}")

        gas_df[f"ma_{ma_window}_{i}"]  = gas_df_raw[i].rolling(ma_window).mean()

    return gas_df, new_columns_list

def init_gas_analysys(gas_df_raw, gas_sensor, range_ppb, backup_folder, df_pickle_path, output_folder, target_value, features_list, debug_mode):
    gas_df = gas_df_raw.copy()

    print(gas_df.head())

    debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
    debug("---------------------- REALIZANDO ANALISE ESTATISTICA ----------------------", "cyan", debug_mode)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)
    
    gas_df_processed = remove_invalid_data(gas_df, range_ppb, gas_sensor, output_folder, target_value, features_list, debug_mode)

    debug(f"Salvando dataset no arquivo de backup {df_pickle_path}", "cyan", debug_mode)

    if not os.path.exists(backup_folder): os.makedirs(backup_folder)
    gas_df_processed.to_pickle(df_pickle_path)

    debug("Sucesso!", "green", debug_mode)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    return gas_df_processed

def main():
    debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
    debug("----------------------- INICIALIZACAO DO EXPERIMENTO -----------------------", "cyan", debug_mode)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    config      = load_config(config_path, debug_mode)
    gas_options = config["gas_name"]

    print("\nSelecao do gas para analise:\n")

    i = 1
    for key in gas_options:
        description = gas_options[key]

        print(f"{i} - {key} ({description})")
        i += 1
    print(f"{i} - Sair")
    
    option = get_number_by_range(1, i)

    if option == i: sys.exit(0)

    gas_sensor = list(gas_options.items())[option - 1][0]
    gas_name   = gas_options[gas_sensor]

    debug(f"\nGAS SELECIONADO: {gas_sensor} ({gas_name})", "green", True)

    range_ppb      = config["gas_range_ppb"][gas_sensor]
    backup_folder  = "backup_files"
    output_folder  = f"output/{gas_sensor}"
    df_pickle_path = "{}/{}_df_pickle".format(backup_folder, gas_sensor)
    bkp_available  = os.path.exists(df_pickle_path)

    if not os.path.exists("output/"):     os.makedirs("output/")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    gas_df = load_dataframe(config, gas_sensor, df_pickle_path, bkp_available)
    gas_df = pd.concat([gas_df, set_datatime_col(gas_df)], axis=1, join='inner')

    target_value  = config["target_value"]
    features_list = list(config["features_data"].keys())

    gas_analysis = Gas_Analysis(gas_df, config, gas_sensor, debug_mode)
    gas_df = gas_analysis.gas_df

    apply_ma = confirm_info("Deseja realizar a analise considerando as medias moveis destacadas no arquivo de configuracao?")

    if apply_ma:
        ma_windows_list = config["moving_average_window"]

        i = 0
        while i < len(ma_windows_list):
            description = ma_windows_list[i]

            print(f"{i+1} - {description}")
            i += 1
        
        ma_window               = ma_windows_list[get_number_by_range(1, i) - 1]
        ma_df, new_columns_list = apply_moving_average(gas_df, ma_window, target_value, features_list)
        
        gas_df        = pd.concat([gas_df, ma_df], axis=1, join='inner')
        target_value  = new_columns_list[-1]
        features_list = new_columns_list[:-1]

    gas_df = gas_df.dropna().copy()
    gas_df = init_gas_analysys(gas_df, gas_sensor, range_ppb, backup_folder, df_pickle_path, output_folder, target_value, features_list, debug_mode)

    get_comparsion_graph(gas_df, gas_sensor, target_value, features_list, output_folder)

    input_text   = "Selecao do algoritmo de analise:\n"
    options_list = ["1) Regressoes lineares", "2) K-Nearest Neighbors", "3) Random Forest", "4) Support Vector Machine", "5) Redes Neurais Artificiais", "6) K-Nearest Regression", "7) Support Vector Regression", "8) Sair"]

    sts_path = f"{output_folder}/statistics_processed_{gas_sensor}.json"
    sts      = open_json_file(sts_path)

    values_list             = gas_df[target_value].tolist()
    categorical_values_list = []

    for v in values_list:
        if   v >= sts["75%"]: categorical_values_list.append(3) #alert
        elif v >= sts["50%"]: categorical_values_list.append(2) #high
        elif v >= sts["25%"]: categorical_values_list.append(1) #moderate
        else:                 categorical_values_list.append(0) #low

    gas_df["categorical_values"] = categorical_values_list
    
    climatic_element = "_".join(features_list)

    debug("Amostra de dados do dataset:\n", "yellow", debug_mode)
    debug(gas_df.head(), "cyan", debug_mode)

    print(input_text)
    for i in options_list:  
        print(i)

    option = get_number_by_range(1, len(options_list))

    if option == 8:
        sys.exit(0)

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
            
            # features_list         = ["rh_value"]
            features_list         = ["temp_value"]
            # target_value = "Value"

            window_size = 60
            # x = gas_df[f'ma_{window_size}_temp'].tolist()
            # y = gas_df[f'ma_{window_size}_gas'].tolist()
            x = gas_df[features_list].values
            y = gas_df[target_value].values
            # x = gas_df['rh_value'].tolist()

            climatic_element = "_".join(features_list)

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

            output_slr_path = f"{output_folder}/simple_linear_regression"

            if not os.path.exists(output_slr_path): os.makedirs(output_slr_path)

            debug(f"\nSalvando metricas no diretorio: {output_slr_path}", "cyan", debug_mode)

            dict_to_json_file(output_slr_path + "/metrics.json", slr_dict)
            dict_to_json_file(output_slr_path + f"/metrics_{gas_sensor}_{climatic_element}.json", slr_dict)

            debug("SUCESSO!", "green", debug_mode)

            fig, ax1 = plt.subplots()
            x_label  = "Temperatura [ºC]" if climatic_element == "temp_value" else "RH [%]" if climatic_element == "rh_value" else "feature variation"

            ax1.set_title(f"Reta de regressao ({gas_sensor} [ppm] - {x_label})", size=25)
            ax1.plot(X_test, y_pred, color="blue", linewidth=3)
            ax1.scatter(X_test, y_test, color="black")
            ax1.set_xlabel(x_label, size=25)
            ax1.set_ylabel(f"Concentracao - {gas_sensor} [ppm]", size=25)
            plt.grid()
            plt.show()

            fig.tight_layout()

            fig.savefig('{}/{}_{}_{}.png'.format(output_slr_path, "slr", "regression_line", climatic_element), dpi=fig.dpi)

        if lin_reg_option == 2:
            debug("INICIALIZANDO ANALISE POR REGRESSAO LINEAR MULTIPLA", debug=debug_mode)
            
            # features_list         = ["temp_value", "rh_value"]
            # target_value = "Value"

            climatic_element = "_".join(features_list)

            x = gas_df[features_list].values
            y = gas_df[target_value].values
            
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

            output_mlr_path = f"{output_folder}/multiple_linear_regression"

            if not os.path.exists(output_mlr_path): os.makedirs(output_mlr_path)

            debug(f"\nSalvando metricas no diretorio: {output_mlr_path}", "cyan", debug_mode)

            dict_to_json_file(output_mlr_path + f"/metrics_{gas_sensor}_{climatic_element}.json", mlr_dict)

            debug("SUCESSO!", "green", debug_mode)
    if option == 2:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("---------------------------- K-NEAREST NEIGHBORS ---------------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        x = gas_df[features_list].values
        y = gas_df["categorical_values"].values

        neighbors         = 25
        neighbors         = np.arange(1,neighbors)
        test_accuracy_knn = np.empty(len(neighbors))

        knn_accuracy_list         = []
        k_variation 	          = []

        max_tries = 10
        i         = 0
        while i < max_tries:
            debug(f"INCIANDO LACO {i}\n", "yellow", debug_mode)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            for j,k in enumerate(neighbors):
                knn = KNeighborsClassifier(n_neighbors=k)
            
                #divide o modelo
                knn.fit(X_train, y_train)	

                # Calcula a acuracia dos dados de teste
                test_accuracy_knn[j] = knn.score(X_test, y_test)

            knn_accuracy_list.append(max(test_accuracy_knn))
            k_variation.append(np.argmax(test_accuracy_knn) + 1)

            i += 1

        k_variation      = np.array(k_variation)
        k_variation_mean = np.mean(k_variation)

        knn_accuracy_list   = np.array(knn_accuracy_list)
        knn_accuracy_mean   = np.mean(knn_accuracy_list)
        
        print("Acuracia media: " + str(knn_accuracy_mean))

        print("-------Vizinhos-------")
        print("Quantidade de vizinhos com melhor acuracia em cada execucao: " + str(k_variation))
        print("Media: " + str(k_variation_mean))
        print("Min:   " + str(np.amin(k_variation)))
        print("Max:   " + str(np.amax(k_variation)))

        most_repeated_k = np.bincount(k_variation).argmax()
        print("numero de vizinhos mais repetido: " + str(most_repeated_k))

        # Verificando precisao do K-NN e Random Forest para diferentes valores de K e arvores de decisao
        fig, ax1 = plt.subplots()

        ax1.set_title('k-NN variando numero de vizinhos proximos', size=25)
        ax1.plot(neighbors, test_accuracy_knn, label='Acuracia K-NN')
        ax1.set_xlabel('Numero de vizinhos proximos', size=25)
        ax1.set_ylabel('Acuracia', size=25)
        plt.grid()
        plt.show()

        fig.tight_layout()

        output_knn_path = f"{output_folder}/knn"

        if not os.path.exists(output_knn_path): os.makedirs(output_knn_path)

        fig.savefig('{}/{}_{}.png'.format(output_knn_path, "knn", "variation"), dpi=fig.dpi)

    ###########################################################################################
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        knn_classifier = KNeighborsClassifier(n_neighbors=most_repeated_k)
        knn_classifier.fit(X_train, y_train)

        knn_y_pred = knn_classifier.predict(X_test)

        print("---------------------------------")
        print("Matriz de confusao")
        print(confusion_matrix(y_test, knn_y_pred))

        print("---------------------------------")
        print("Cross tab")
        print(pd.crosstab(y_test, knn_y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        print("---------------------------------")
        print("Diagnostico de classificacao")
        print(classification_report(y_test,knn_y_pred))

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=knn_y_pred)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predicao (classificacao)', fontsize=18)
        plt.ylabel('Valor real', fontsize=18)
        plt.title('Matriz de confusao - KNN', fontsize=18)
        plt.show()
        fig.savefig(f"{output_knn_path}/knn_confusion_matrix.png")

        knn         = {}
        knn["r2"]   = get_r2_score(y_test, knn_y_pred)
        knn["mse"]  = get_MSE(y_test, knn_y_pred)
        knn["rmse"] = get_RMSE(y_test, knn_y_pred)
        knn["mae"]  = get_MAE(y_test, knn_y_pred)

        debug(f'R^2:  {knn["r2"]}', debug=debug_mode)
        debug(f'MSE:  {knn["mse"]}', debug=debug_mode)
        debug(f'RMSE: {knn["rmse"]}', debug=debug_mode)
        debug(f'MAE:  {knn["mae"]}', debug=debug_mode)
        
        debug(f"\nSalvando metricas no diretorio: {output_knn_path}", "cyan", debug_mode)

        dict_to_json_file(output_knn_path + f"/metrics_{gas_sensor}_{climatic_element}.json", knn)

        debug("SUCESSO!", "green", debug_mode)
    if option == 3:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("------------------------------- RANDOM FOREST ------------------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        x = gas_df[features_list].values
        y = gas_df["categorical_values"].values

        max_trees                   = 25
        trees                       = np.arange(1,max_trees)
        test_accuracy_random_forest = np.empty(len(trees))

        rf_accuracy_list = []
        trees_variation  = []

        max_tries = 10
        i         = 0
        while i < max_tries:
            debug(f"INCIANDO LACO {i}\n", "yellow", debug_mode)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            for j,k in enumerate(trees):
                random_forest = RandomForestClassifier(n_estimators=k)  

                #divide o modelo
                random_forest.fit(X_train, y_train)

                # Calcula a precisao dos dados de teste
                test_accuracy_random_forest[j] = random_forest.score(X_test, y_test)

            rf_accuracy_list.append(max(test_accuracy_random_forest))
            trees_variation.append(np.argmax(test_accuracy_random_forest))

            i += 1

        trees_variation     = np.array(trees_variation)
        tree_variation_mean = np.mean(trees_variation)

        rf_accuracy_list  = np.array(rf_accuracy_list)
        rf_accuracy_mean  = np.mean(rf_accuracy_list)

        print("Acuracia media: " + str(rf_accuracy_mean))

        print("-------Arvores-------")
        print("Quantidade de arvores com melhor acuracia em cada execucao: " + str(trees_variation))
        print("Media: " + str(tree_variation_mean))
        print("Min:   " + str(np.amin(trees_variation)))
        print("Max:   " + str(np.amax(trees_variation)))

        most_repeated_trees = np.bincount(trees_variation).argmax()
        print("numero de arvores mais repetido: " + str(most_repeated_trees))

       # Verificando acuracia do Random Forest para diferentes valores de arvores de decisao
        fig, ax1 = plt.subplots()

        ax1.set_title('Random Forest variando numero arvores de decisao', size=25)
        ax1.plot(trees, test_accuracy_random_forest, label='Acuracia Random Forest')
        ax1.set_xlabel('Numero de arvores de decisao', size=25)
        ax1.set_ylabel('Acuracia', size=25)
        plt.grid()
        plt.show()

        fig.tight_layout()

        output_rf_path = f"{output_folder}/random_forest"

        if not os.path.exists(output_rf_path): os.makedirs(output_rf_path)

        fig.savefig('{}/{}_{}.png'.format(output_rf_path, "trees", "variation"), dpi=fig.dpi)

    ###########################################################################################
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        rf_classifier = RandomForestClassifier(n_estimators=most_repeated_trees)
        rf_classifier.fit(X_train, y_train)

        rf_y_pred = rf_classifier.predict(X_test)

        print("---------------------------------")
        print("Matriz de confusao")
        print(confusion_matrix(y_test, rf_y_pred))

        print("---------------------------------")
        print("Cross tab")
        print(pd.crosstab(y_test, rf_y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        print("---------------------------------")
        print("Diagnostico de classificacao")
        print(classification_report(y_test,rf_y_pred))

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=rf_y_pred)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predicao (classificacao)', fontsize=18)
        plt.ylabel('Valor real', fontsize=18)
        plt.title('Matriz de confusao - Random Forest', fontsize=18)
        plt.show()
        fig.savefig(f"{output_rf_path}/rf_confusion_matrix.png")

        debug("\nObtendo metricas", "cyan", debug_mode)

        rf         = {}
        rf["r2"]   = get_r2_score(y_test, rf_y_pred)
        rf["mse"]  = get_MSE(y_test, rf_y_pred)
        rf["rmse"] = get_RMSE(y_test, rf_y_pred)
        rf["mae"]  = get_MAE(y_test, rf_y_pred)

        debug(f'R^2:  {rf["r2"]}', debug=debug_mode)
        debug(f'MSE:  {rf["mse"]}', debug=debug_mode)
        debug(f'RMSE: {rf["rmse"]}', debug=debug_mode)
        debug(f'MAE:  {rf["mae"]}', debug=debug_mode)
        
        debug(f"\nSalvando metricas no diretorio: {output_rf_path}", "cyan", debug_mode)

        dict_to_json_file(output_rf_path + f"/metrics_{gas_sensor}_{climatic_element}.json", rf)

        debug("SUCESSO!", "green", debug_mode)
    if option == 4:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("--------------------------- SUPPORT VECTOR MACHINE -------------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        x = gas_df[features_list].values
        y = gas_df["categorical_values"].values

        svm_accuracy_list = []

        max_tries = 10
        i         = 0
        while i < max_tries:
            debug(f"INCIANDO LACO {i}\n", "yellow", debug_mode)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            svm_classifier = SVC(kernel="linear")
            svm_classifier.fit(X_train, y_train)

            svm_accuracy_list.append(svm_classifier.score(X_test, y_test))

            i += 1

        svm_accuracy_list = np.array(svm_accuracy_list)
        svm_accuracy_mean = np.mean(svm_accuracy_list)

        print("Acuracia media: " + str(svm_accuracy_mean))

        output_svm_path = f"{output_folder}/svm"

        if not os.path.exists(output_svm_path): os.makedirs(output_svm_path)
    ###########################################################################################
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        svm_classifier = SVC(kernel="linear")
        svm_classifier.fit(X_train, y_train)

        # from mlxtend.plotting import plot_decision_regions

        # print(y_train)

        # # Plotting decision regions
        # plot_decision_regions(X_train, y_train, clf=svm_classifier, legend=2)
        # plt.show()

        svm_y_pred = svm_classifier.predict(X_test)

        print("---------------------------------")
        print("Matriz de confusao")
        print(confusion_matrix(y_test, svm_y_pred))

        print("---------------------------------")
        print("Cross tab")
        print(pd.crosstab(y_test, svm_y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        print("---------------------------------")
        print("Diagnostico de classificacao")
        print(classification_report(y_test,svm_y_pred))

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=svm_y_pred)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predicao (classificacao)', fontsize=18)
        plt.ylabel('Valor real', fontsize=18)
        plt.title('Matriz de confusao - SVM', fontsize=18)
        plt.show()
        fig.savefig(f"{output_svm_path}/svm_confusion_matrix.png")

        debug("\nObtendo metricas", "cyan", debug_mode)

        svm         = {}
        svm["r2"]   = get_r2_score(y_test, svm_y_pred)
        svm["mse"]  = get_MSE(y_test, svm_y_pred)
        svm["rmse"] = get_RMSE(y_test, svm_y_pred)
        svm["mae"]  = get_MAE(y_test, svm_y_pred)

        debug(f'R^2:  {svm["r2"]}', debug=debug_mode)
        debug(f'MSE:  {svm["mse"]}', debug=debug_mode)
        debug(f'RMSE: {svm["rmse"]}', debug=debug_mode)
        debug(f'MAE:  {svm["mae"]}', debug=debug_mode)
        
        debug(f"\nSalvando metricas no diretorio: {output_svm_path}", "cyan", debug_mode)

        dict_to_json_file(output_svm_path + f"/metrics_{gas_sensor}_{climatic_element}.json", svm)

        debug("SUCESSO!", "green", debug_mode)
    if option == 5:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("-------------------------- REDES NEURAIS ARTIFICIAIS -----------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        x = gas_df[features_list].values
        y = gas_df["categorical_values"].values

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        input_layer_size        = len(features_list)                                               # camada de entrada, igual a quantidade de features consideradas na analise
        output_dense_layer_size = 4                                                                # camada de saida, igual a quantidade de classes
        # dense_layer_size        = pow(2,(math.ceil(input_layer_size + output_dense_layer_size)/2)) # quantidade de neuronios na camada densa
        dense_layer_size        = 64                                                               # quantidade de neuronios na camada densa
        n_epochs                = 30                                                               # quantiddade de vezes que o treinamento ira acontecer

        output_ann_path = f"{output_folder}/ann"

        if not os.path.exists(output_ann_path): os.makedirs(output_ann_path)

        debug("Ajustando pesos das classes...", "blue", debug_mode)

        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        weights = {0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3]}

        debug(f"Pesos ajustados: {weights}", "green", debug_mode)

        model = Sequential()
        model.add(InputLayer(input_shape=(input_layer_size,)))
        model.add(Dense(dense_layer_size, activation= "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(output_dense_layer_size, activation="softmax"))
        
        debug("Estrutura da Rede Neural", "cyan", debug_mode)
        model.summary()
        plot_model(model, to_file=f"{output_ann_path}/ann_model.png")

        cp           = ModelCheckpoint(output_ann_path, save_best_only=True) # Salva somente a melhor execucao
        metrics_list = [RootMeanSquaredError(), MeanAbsoluteError(), MeanSquaredError(), "accuracy"]

        debug("Compilando modelo da Rede Neural...", "blue", debug_mode)
        # model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=metrics_list)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=metrics_list)

        debug("Sucesso!", "green", debug_mode)
        debug("Realizando treinamento...", "blue", debug_mode)

        model.fit(x=X_train, y=y_train, epochs=n_epochs, validation_data=(X_test, y_test), callbacks=[cp], class_weight=weights)

        debug("Sucesso!", "green", debug_mode)
        debug("Carregando modelo com melhor resultado obtido...", "blue", debug_mode)

        ann_model = load_model(output_ann_path)

        debug("Sucesso!", "green", debug_mode)
        debug("Extraindo metricas de desempenho...", "blue", debug_mode)

        ann_y_pred = np.argmax(ann_model.predict(X_test), axis=-1)

        print("\n---------------------------------")
        print("Matriz de confusao")
        print(confusion_matrix(y_test, ann_y_pred))

        print("---------------------------------")
        print("Cross tab")
        print(pd.crosstab(y_test, ann_y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        print("---------------------------------")
        print("Diagnostico de classificacao")
        print(classification_report(y_test, ann_y_pred))

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=ann_y_pred)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predicao (classificacao)', fontsize=18)
        plt.ylabel('Valor real', fontsize=18)
        plt.title('Matriz de confusao - Rede Neural Artificial', fontsize=18)
        plt.show()
        fig.savefig(f"{output_ann_path}/ann_confusion_matrix.png")

        debug("\nObtendo metricas", "cyan", debug_mode)

        ann         = {}
        ann["r2"]   = get_r2_score(y_test, ann_y_pred)
        ann["mse"]  = get_MSE(y_test, ann_y_pred)
        ann["rmse"] = get_RMSE(y_test, ann_y_pred)
        ann["mae"]  = get_MAE(y_test, ann_y_pred)

        debug(f'R^2:  {ann["r2"]}', debug=debug_mode)
        debug(f'MSE:  {ann["mse"]}', debug=debug_mode)
        debug(f'RMSE: {ann["rmse"]}', debug=debug_mode)
        debug(f'MAE:  {ann["mae"]}', debug=debug_mode)
        
        debug(f"\nSalvando metricas no diretorio: {output_ann_path}", "cyan", debug_mode)

        dict_to_json_file(output_ann_path + f"/metrics_{gas_sensor}_{climatic_element}.json", ann)

        debug("SUCESSO!", "green", debug_mode)
    if option == 6:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("---------------------------- K-NEAREST REGRESSION---------------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        x = gas_df[features_list].values
        y = gas_df[target_value].values

        # knr_weight        = "distance"
        knr_weight        = "uniform"
        neighbors         = 25
        neighbors         = np.arange(1,neighbors)
        test_accuracy_knr = np.empty(len(neighbors))

        knr_accuracy_list         = []
        k_variation 	          = []

        debug("Iniciando calibracao da quantidade de vizinhos proximos", "cyan", debug_mode)

        max_tries = 10
        i         = 0
        while i < max_tries:
            debug(f"INCIANDO LACO {i}\n", "yellow", debug_mode)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            for j,k in enumerate(neighbors):
                knr = KNeighborsRegressor(n_neighbors=k, weights=knr_weight)
            
                #divide o modelo
                knr.fit(X_train, y_train)

                # Calcula a acuracia dos dados de teste
                test_accuracy_knr[j] = knr.score(X_test, y_test)

            knr_accuracy_list.append(max(test_accuracy_knr))
            k_variation.append(np.argmax(test_accuracy_knr) + 1)

            i += 1

        k_variation      = np.array(k_variation)
        k_variation_mean = np.mean(k_variation)

        knr_accuracy_list = np.array(knr_accuracy_list)
        knr_accuracy_mean = np.mean(knr_accuracy_list)
        
        print("Acuracia media: " + str(knr_accuracy_mean))

        print("-------Vizinhos-------")
        print("Quantidade de vizinhos com melhor acuracia em cada execucao: " + str(k_variation))
        print("Media: " + str(k_variation_mean))
        print("Min:   " + str(np.amin(k_variation)))
        print("Max:   " + str(np.amax(k_variation)))

        most_repeated_k = np.bincount(k_variation).argmax()
        print("numero de vizinhos mais repetido: " + str(most_repeated_k))

        output_knr_path = f"{output_folder}/knr"

        if not os.path.exists(output_knr_path): os.makedirs(output_knr_path)

        fig, ax1 = plt.subplots()

        ax1.set_title('k-NR variando numero de vizinhos proximos', size=25)
        ax1.plot(neighbors, test_accuracy_knr, label='Acuracia K-NN')
        ax1.set_xlabel('Numero de vizinhos proximos', size=25)
        ax1.set_ylabel('Acuracia', size=25)
        plt.grid()
        plt.show()

        fig.tight_layout()

        fig.savefig('{}/{}_{}_{}.png'.format(output_knr_path, "knr", "variation", climatic_element), dpi=fig.dpi)

    ###########################################################################################
        debug("\nSeparando dados de treinamento e teste...", "cyan", debug_mode)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        debug("\nCriando modelo de regressao com dados de treinamento", "cyan", debug_mode)

        knr_classifier = KNeighborsRegressor(n_neighbors=most_repeated_k, weights=knr_weight)
        knr_classifier.fit(X_train, y_train)

        debug(knr_classifier.get_params())

        debug("\nObtendo dados de teste", "cyan", debug_mode)

        knr_y_pred = knr_classifier.predict(X_test)

        if len(features_list) == 1:
            fig, ax1 = plt.subplots()

            ax1.set_title("KNeighborsRegressor (k = %i, weights = '%s')" % (most_repeated_k, knr_weight), size=25)
            ax1.scatter(X_train, y_train, color="darkorange", label="Dados de treinamento")
            ax1.scatter(X_test, knr_y_pred, color="navy", label="Predicao")
            x_label = "Temperatura [ºC]" if climatic_element == "temp_value" else "RH [%]" if climatic_element == "rh_value" else "feature variation"
            ax1.set_xlabel(x_label, size=25)
            ax1.set_ylabel(f"Concentracao - {gas_sensor} [ppm]", size=25)
            plt.grid()
            plt.tight_layout()
            plt.show()

            fig.savefig('{}/{}_{}_{}.png'.format(output_knr_path, "knr", "prediction", climatic_element), dpi=fig.dpi)

        debug("\nObtendo metricas", "cyan", debug_mode)

        knr         = {}
        knr["r2"]   = get_r2_score(y_test, knr_y_pred)
        knr["mse"]  = get_MSE(y_test, knr_y_pred)
        knr["rmse"] = get_RMSE(y_test, knr_y_pred)
        knr["mae"]  = get_MAE(y_test, knr_y_pred)

        debug(f'R^2:  {knr["r2"]}', debug=debug_mode)
        debug(f'MSE:  {knr["mse"]}', debug=debug_mode)
        debug(f'RMSE: {knr["rmse"]}', debug=debug_mode)
        debug(f'MAE:  {knr["mae"]}', debug=debug_mode)
        
        debug(f"\nSalvando metricas no diretorio: {output_knr_path}", "cyan", debug_mode)

        dict_to_json_file(output_knr_path + f"/metrics_{gas_sensor}_{climatic_element}.json", knr)

        debug("SUCESSO!", "green", debug_mode)
    if option == 7:
        debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
        debug("-------------------------- SUPPORT VECTOR REGRESSION -----------------------", "cyan", debug_mode)
        debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

        x = gas_df[features_list].values
        y = gas_df[target_value].values

        output_svr_path = f"{output_folder}/svr"

        if not os.path.exists(output_svr_path): os.makedirs(output_svr_path)

        debug("\nCalibrando parametro de regularizacao (C)...", "cyan", debug_mode)

        N                        = 500
        c_array                  = np.linspace(0.1, 5, N)
        svr_epsilon              = 20
        svr_kernel_func          = "rbf"
        test_mae_list            = []
        perc_within_epsilon_list = []

        for c in c_array:
            # debug(f"Aplicando C = {c}")
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            svr_regressor = SVR(kernel=svr_kernel_func, C=c, epsilon=svr_epsilon) if svr_kernel_func != "linear" else LinearSVR(C=c, epsilon=svr_epsilon)
            svr_regressor.fit(X_train, y_train)
            svr_y_pred = svr_regressor.predict(X_test)

            test_mae        = get_MAE(y_test, svr_y_pred)
            perc_within_eps = 100*np.sum(abs(y_test-svr_y_pred) <= svr_epsilon) / len(y_test)

            test_mae_list.append(test_mae)
            perc_within_epsilon_list.append(perc_within_eps)

        m     = min(test_mae_list)
        inds  = [i for i, j in enumerate(test_mae_list) if j == m]
        svr_C = c_array[inds[0]]

        debug(f"Melhor valor encontrado! C = {svr_C}", "green", debug_mode)

        fig, ax1 = plt.subplots(figsize=(12,7))
        ax1.set_title(f"Analise do parametro C para Epsilon = {svr_epsilon}", size=25)

        color='green'
        ax1.set_xlabel('variacao de C')
        ax1.set_ylabel('% de dados dentro das margens de Epsilon', color=color)
        ax1.scatter(c_array, perc_within_epsilon_list, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.grid(color=color, which='both')

        color='blue'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Erro absoluto medio (MAE)', color=color)  # we already handled the x-label with ax1
        ax2.scatter(c_array, test_mae_list, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.grid(color=color, which='both')
        plt.show()

        fig.savefig('{}/{}_{}_{}_{}.png'.format(output_svr_path, "svr", "set_C", svr_epsilon, climatic_element), dpi=fig.dpi)

        debug("\nSeparando dados de treinamento e teste...", "cyan", debug_mode)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        debug("\nCriando modelo de regressao com dados de treinamento", "cyan", debug_mode)

        svr_regressor = SVR(kernel=svr_kernel_func, C=svr_C, epsilon=svr_epsilon) if svr_kernel_func != "linear" else LinearSVR(C=svr_C, epsilon=svr_epsilon)
        svr_regressor.fit(X_train, y_train)

        debug(svr_regressor.get_params(), )
        debug("\nObtendo dados de teste", "cyan", debug_mode)

        svr_y_pred = svr_regressor.predict(X_test)

        if len(features_list) == 1:
            svr_C    = "%.2f" % svr_C
            fig, ax1 = plt.subplots()

            ax1.set_title(f"Suppor Vector Regressor (C = {svr_C}, Epsilon = {svr_epsilon})", size=25)
            ax1.scatter(X_train, y_train, color="green", label="Dados de treinamento")

            if svr_kernel_func != "linear":
                ax1.plot(X_test, svr_y_pred, color='navy')
                ax1.plot(X_test, svr_y_pred+svr_epsilon, color='darkorange')
                ax1.plot(X_test, svr_y_pred-svr_epsilon, color='darkorange')
            else:
                ax1.scatter(X_test, svr_y_pred, color='navy')
                ax1.scatter(X_test, svr_y_pred+svr_epsilon, color='darkorange')
                ax1.scatter(X_test, svr_y_pred-svr_epsilon, color='darkorange')

            x_label = "Temperatura [ºC]" if climatic_element == "temp_value" else "RH [%]" if climatic_element == "rh_value" else "feature variation"
            ax1.set_xlabel(x_label, size=25)
            ax1.set_ylabel(f'Concentracao [{gas_sensor}] - ppm', size=25)
            plt.grid()
            plt.tight_layout()
            plt.show()

            fig.savefig('{}/{}_{}_{}.png'.format(output_svr_path, "svr", "prediction", climatic_element), dpi=fig.dpi)

        debug("\nObtendo metricas", "cyan", debug_mode)

        svr         = {}
        svr["r2"]   = get_r2_score(y_test, svr_y_pred)
        svr["mse"]  = get_MSE(y_test, svr_y_pred)
        svr["rmse"] = get_RMSE(y_test, svr_y_pred)
        svr["mae"]  = get_MAE(y_test, svr_y_pred)

        if svr_kernel_func == "linear":
            svr["coefficients"]              = {}
            svr["coefficients"]["intercept"] = svr_regressor.intercept_.tolist()
            svr["coefficients"]["coefs"]     = svr_regressor.coef_.tolist()

            debug(f"Intercept:   {svr_regressor.intercept_}", debug=debug_mode)
            debug(f"Coefficient: {svr_regressor.coef_}", debug=debug_mode)

        debug(f'R^2:  {svr["r2"]}', debug=debug_mode)
        debug(f'MSE:  {svr["mse"]}', debug=debug_mode)
        debug(f'RMSE: {svr["rmse"]}', debug=debug_mode)
        debug(f'MAE:  {svr["mae"]}', debug=debug_mode)
        
        debug(f"\nSalvando metricas no diretorio: {output_svr_path}", "cyan", debug_mode)

        dict_to_json_file(output_svr_path + f"/metrics_{gas_sensor}_{climatic_element}.json", svr)

        debug("SUCESSO!", "green", debug_mode)

def remove_invalid_data(gas_df, range_ppb, gas_sensor, output_folder, target_value, features_list, debug_mode):
    gas_value = gas_df[target_value]

    debug("Dados sem processamento...", "cyan", debug_mode)
    statistics_raw = get_statistics_data(gas_value, range_ppb)
 
    debug(f"\nCalculando correlacao com as features {features_list}", "blue", debug_mode)

    for i in features_list:
        feature_value    = gas_df[i]
        corr_gas_feature = get_correlation(gas_value, feature_value)

        statistics_raw[f'corr_gas_{i}'] = corr_gas_feature

    # pprint.pprint(statistics_raw)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    debug("Dados processados...", "cyan", debug_mode)

    debug("\nRemovendo dados fora dos limiares do sensor {}".format(gas_sensor), "blue", debug_mode)

    gas_df_processed = remove_values_outside_range(gas_df, target_value, range_ppb)
    gas_value_proc   = gas_df_processed[target_value]

    statistics_proc = get_statistics_data(gas_value_proc, range_ppb)

    for i in features_list:
        feature_value    = gas_df_processed[i]
        corr_gas_feature = get_correlation(gas_value_proc, feature_value)

        statistics_proc[f'corr_gas_{i}'] = corr_gas_feature

    # pprint.pprint(statistics_proc)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    debug("Removendo outliers".format(gas_sensor), "blue", debug_mode)

    gas_df_processed_out = remove_outliers_from_df(gas_df_processed, target_value, statistics_proc["lower_thr"], statistics_proc["upper_thr"])
    gas_value_out        = gas_df_processed_out[target_value]

    statistics_processed = get_statistics_data(gas_value_out, range_ppb)

    for i in features_list:
        feature_value    = gas_df_processed_out[i]
        corr_gas_feature = get_correlation(gas_value_out, feature_value)

        statistics_processed[f'corr_gas_{i}'] = corr_gas_feature

    # pprint.pprint(statistics_processed)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    output_file_raw       = "{}/statistics_raw_{}.json".format(output_folder, gas_sensor)
    output_file_processed = "{}/statistics_processed_{}.json".format(output_folder, gas_sensor)
    
    dict_to_json_file(output_file_raw, statistics_raw)
    dict_to_json_file(output_file_processed, statistics_processed)

    return gas_df_processed_out

def get_statistics_data(gas_value, range_ppb):
    debug("\nContando dados fora dos limiares", "blue", debug_mode)
    greatter_than_range_count, less_than_zero_count = count_values_outside_range(gas_value, range_ppb)

    debug("\nCalculando metricas", "blue", debug_mode)
    statistics = get_statistics_by_describe(gas_value)

    debug("\nCalculando outliers", "blue", debug_mode)
    outliers_info = get_outliers(gas_value, statistics["25%"], statistics["75%"])

    debug("\nSucesso! Registrando informacoes...", "green", debug_mode)
    statistics['greatter_than_range_count'] = int(greatter_than_range_count)
    statistics['less_than_zero_count']      = int(less_than_zero_count)
    statistics = {** statistics, **outliers_info}

    return statistics

if __name__ == "__main__":
    while True:
        main()