from load import load_data

def split_training_data() :
    """
    Getting the previous weeks.

    Returns:
        Data divided in training and testing data.
    """
    data = load_data()

    data["naive_pred"] = data["cases"].shift(52)

    split_point_one = int(len(data) * 0.6)
    split_point_two = int(len(data) * 0.8)


    data_train = data.iloc[:split_point_one]
    data_val = data.iloc[split_point_one:split_point_two]
    data_test = data.iloc[split_point_two:]

    return data_train, data_val ,data_test


