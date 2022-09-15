from scipy.stats  import pearsonr

def count_values_outside_range(gas_value, range_ppb):
    greatter_than_range_count = (gas_value > range_ppb).sum()
    less_than_zero_count      = (gas_value < 0).sum()

    return greatter_than_range_count, less_than_zero_count

def remove_values_outside_range(input_gas_df, gas_value, range_ppb):
    gas_df = input_gas_df.copy()

    # Get outside values range to exclude the correct indexes from dataset
    greatter_than_range_index = gas_df[gas_value > range_ppb].index.tolist()
    less_than_range_index     = gas_df[gas_value < 0].index.tolist()

    # Remove values outside the sensor range
    gas_df = gas_df.drop(gas_df.index[greatter_than_range_index])
    gas_df = gas_df.drop(gas_df.index[less_than_range_index])

    return gas_df

# Get count, mean, std, min, 25%, 50%, 75%, max and variance values from df
def get_statistics_by_describe(gas_value):
    statistics        = gas_value.describe().to_dict()
    statistics['var'] = gas_value.var()

    return statistics

def get_outliers(gas_value, first_quartile, third_quartile):
    # Calc for Outliers
    lower_thr = first_quartile - 1.5 * (third_quartile - first_quartile)
    upper_thr = third_quartile + 1.5 * (third_quartile - first_quartile)

    outiler_lower_count = (gas_value < lower_thr).sum()
    outiler_upper_count = (gas_value > upper_thr).sum()

    outliers_info = {}
    outliers_info['lower_thr']           = lower_thr
    outliers_info['upper_thr']           = upper_thr
    outliers_info['outiler_lower_count'] = int(outiler_lower_count)
    outliers_info['outiler_upper_count'] = int(outiler_upper_count)

    return outliers_info

def get_correlation(data1_df, data2_df):
    data1_list = data1_df.tolist()
    data2_list = data2_df.tolist()

    correlation, _ = pearsonr(data1_list, data2_list)

    return correlation

def remove_outliers_from_df(input_gas_df, gas_value, lower_thr, upper_thr):
    gas_df = input_gas_df.copy()

    # Get outliers list to exclude the correct indexes from dataset
    greatter_than_range_index = gas_df[gas_value > upper_thr].index.tolist()
    less_than_range_index     = gas_df[gas_value < lower_thr].index.tolist()

    # Remove outliers
    gas_df = gas_df.drop(gas_df.index[greatter_than_range_index])
    gas_df = gas_df.drop(gas_df.index[less_than_range_index])

    return gas_df