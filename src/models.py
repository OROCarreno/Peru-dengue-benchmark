from split import split_training_data
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

RANDOM_STATE = 55 ## To keep the same result in trees.



def season_naive():
    """
    Comparing a season naive forecast.
    Returns:
        Mean aboslute error of this season naive forecast.
    """
    seasonNav_data_train,seasonNav_data_val,_ = split_training_data()
    seasonNav_data_train = seasonNav_data_train.dropna(subset="naive_pred")
    print("SEASON NAIVE")
    print(f"MAE trainn: {mean_absolute_error(seasonNav_data_train["cases"], seasonNav_data_train["naive_pred"])} MAE val: {mean_absolute_error(seasonNav_data_val["cases"], seasonNav_data_val["naive_pred"])}")

# print(season_naive())


data_train,data_val,_ = split_training_data()
feature_colum = ["pre_week_1","pre_week_2","pre_week_3","pre_week_4"]
x_train = data_train[feature_colum]
y_train = data_train["cases"]
x_val = data_val[feature_colum]
y_val = data_val["cases"]


def RandomForest():

    min_samples_split_list = [2, 10, 20, 40, 60]
    max_depth_list = [2, 4, 8, 16, 32,None]
    n_estimators_list = [10,50,100,500]

    error_train_min = []
    error_val_min = []

    error_train_depth = []
    error_val_depth = []

    error_train_est = []
    error_val_est = []
    

    for min_sample_split in min_samples_split_list:
        model = RandomForestRegressor(min_samples_split = min_sample_split, 
                                      random_state= RANDOM_STATE).fit(x_train,y_train) 
        prediction_trian = model.predict(x_train)
        prediction_val = model.predict(x_val)
        error_train = mean_absolute_error(y_train,prediction_trian)
        error_val = mean_absolute_error(y_val,prediction_val)
        error_train_min.append(error_train)
        error_val_min.append(error_val)

    #Plotting to find out what value of mininum sample split gives the lowest value for validation data    
    # plot_table('Train x Validation metrics','min sample list','MAE',min_samples_split_list,error_train_min,
    #                    error_val_min,['Train','Validation'])
    ## lowest validation MAE is 10.

    for max_depth_value in max_depth_list:
        model = RandomForestRegressor( max_depth = max_depth_value, 
                                        random_state= RANDOM_STATE).fit(x_train,y_train) 
        prediction_train = model.predict(x_train)
        prediction_val = model.predict(x_val)
        error_train = mean_absolute_error(y_train,prediction_train)
        error_val = mean_absolute_error(y_val,prediction_val)
        error_train_depth.append(error_train)
        error_val_depth.append(error_val)

    #Plotting to find out what value of maximum depth gives the lowest value for validation data    
    # plot_table('Train x Validation metrics','max depth','MAE',max_depth_list,error_train_depth,
    #                error_val_depth,['Train','Validation'])
    # lowest validation MAE is 8.

    for estimators in n_estimators_list:
        model = RandomForestRegressor( n_estimators= estimators,
                                        random_state= RANDOM_STATE).fit(x_train,y_train) 
        prediction_train = model.predict(x_train)
        prediction_val = model.predict(x_val)
        error_train = mean_absolute_error(y_train,prediction_train)
        error_val = mean_absolute_error(y_val,prediction_val)
        error_train_est.append(error_train) 
        error_val_est.append(error_val)

    #Plotting to find out what value of number of estimators gives the lowest value for validation data     
    # plot_table('Train x Validation metrics','estimators list','MAE',n_estimators_list,error_train_min,
    #            error_val_min,['Train','Validation'])
    #lowest number of estimators 50 


    # So using these values
    random_forest_model = RandomForestRegressor(n_estimators = 50,
                                             max_depth = 8, 
                                             min_samples_split = 10).fit(x_train,y_train)
    print("RANDOM FOREST")
    print(f"MAE Train: {mean_absolute_error(random_forest_model.predict(x_train),y_train)} MAE val: {mean_absolute_error(random_forest_model.predict(x_val),y_val)}")


def plot_table(title, xlable, ylabel, n_list, plot1, plot2, legend):
    
    plt.figure() 
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    plt.xticks(ticks = range(len(n_list )),labels=n_list)
    plt.plot(plot1)
    plt.plot(plot2)
    plt.legend(legend)
    plt.show()
    plt.close()

def XGBoost_tree():
    xgb_model = XGBRegressor(n_estimators = 500, learning_rate = 0.05,early_stopping_rounds=10,verbosity = 1, random_state = RANDOM_STATE)
    xgb_model.fit(x_train,y_train, eval_set = [(x_val,y_val)],verbose=True)

    print("XGBoost ")
    print(f"MAE train: {mean_absolute_error(xgb_model.predict(x_train),y_train)}MEA val: {mean_absolute_error(xgb_model.predict(x_val),y_val)}")

# season_naive()
RandomForest()
# XGBoost_tree()