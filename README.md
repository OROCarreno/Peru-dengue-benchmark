# Peru-dengue-benchmark
A benchmark of four models forecasting the weekly dengue cases in countries where it is an endemic. Peru which experienced a outbreaking record in 2023-24 and Malaysia, with an stedier transition.
Models: Season-naive, Linear Regression, Random forest, XGBoost. 
MAE = mean absolute error

Results Peru: 

Model	Test MAE
Linear Regression	836
Random Forest	2,188
XGBoost	2,610
Seasonal-naive	4,427

Results Malaysia:

Linear Regression:79.74647668468978
Random Forest:99.09175955408001
XGBoost tree:102.84014892578125
Season naive:1013.3951612903226



The reference model is season naive. Linear regression would be the best one for predicting in both countries, It can be seen that the MAE for Malaysia is lower in general, this is because the country experienced a stedier transition which allowed the model to forecast with more precission.


Method
- data: Opendengue 
- Features: each of the previous 4 weeks.
- split: 60/20/20. Training, validation, testing

