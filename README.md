# Peru-dengue-benchmark
A benchmakr of four models forecasting the weekly dengue cases in Peru. 
Models: Season-naive, Linear Regression, Random forest, XGBoost. 

Results: 

Model	Test MAE
Linear Regression	836
Random Forest	2,210
XGBoost	2,610
Seasonal-naive	4,427

The reference model is season naive. Linear regression would be the best one by 5.3x.

Why Linear models is the best?

During 2023-24 Peru's record dengue outbreaks with peaks of around 22000 almost double of anything on the training history
which makes the predictions harder for models that cannot extrapolate like trees. 

Method
- data: Opendengue 
- Features: each of the previous 4 weeks.
- split: 60/20/20. Training, validation, testing

