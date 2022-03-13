import pandas            as pd
import matplotlib.pyplot as plt

datasets_path = 'dados_lcqar_07_08_2020'

def main():
    co_df   = pd.read_csv('{}/CO.CSV'.format(datasets_path))
    no2_df  = pd.read_csv('{}/NO2.CSV'.format(datasets_path))
    o3_df   = pd.read_csv('{}/O3.CSV'.format(datasets_path))
    so2_df  = pd.read_csv('{}/SO2.CSV'.format(datasets_path))
    temp_df = pd.read_csv('{}/TMP.CSV'.format(datasets_path))
    humi_df = pd.read_csv('{}/RH.CSV'.format(datasets_path))
    
    co_df   = co_df.loc[:, ['Value']]
    no2_df  = no2_df.loc[:, ['Value']]
    o3_df   = o3_df.loc[:, ['Value']]
    so2_df  = so2_df.loc[:, ['Value']]
    # temp_df = temp_df.loc[:, ['Value']] * 1000
    # temp_df = temp_df['Value'] - temp_df['Value'].mean()
    # humi_df = humi_df.loc[:, ['Value']] * 100
    # humi_df = humi_df['Value'] - humi_df['Value'].mean()
    humi_df = humi_df.loc[:, ['Value']] * co_df['Value']


    ax = plt.gca()
    # plt.plot(co_df['Value'])
    co_df.plot(ax=ax)
    # no2_df.plot(ax=ax)
    # o3_df.plot(ax=ax)
    # so2_df.plot(ax=ax)
    # temp_df.plot(ax=ax)
    humi_df.plot(ax=ax)
    plt.legend(['CO', 'HUMI'])
    # plt.legend(['CO', 'TEMP', 'HUMI'])
    # plt.legend(['CO', 'NO2', 'O3', 'SO2', 'TEMP', 'HUMI'])
    plt.show()



if __name__ == "__main__":
    main()