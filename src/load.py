import matplotlib.pyplot as plt
import pandas as pd

def load_data() -> pd.DataFrame:
    """
    Loading the data from Opendengue.

    Returns:
        Data with the features of: week, cases and 4 previous weeks. 
    """
    data = pd.read_csv(
        "../data/peru_dengue_weekly_national.csv",
        parse_dates=["week"]
    )

    #Checking data.
    assert data["cases"].min() >= 0 
    assert data["week"].is_monotonic_increasing 
    assert not data["week"].duplicated().any() 

    # getting the previous weeks as features

    data["pre_week_1"] = data["cases"].shift(1)
    data["pre_week_2"] = data["cases"].shift(2)
    data["pre_week_3"] = data["cases"].shift(3)
    data["pre_week_4"] = data["cases"].shift(4)

    #Getting rid of the NaN
    data = data.dropna()

    return data


if __name__ == "__main__":
    print(load_data().head())

