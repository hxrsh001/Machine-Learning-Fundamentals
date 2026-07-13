import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Actual Value Prediction-
y_true = np.array([50,60,75,90,100])

# Predicted Values-
y_pred = np.array([52,58,78,88,105])

# MAE (Mean Absolute Error)-
mae = mean_absolute_error(y_true, y_pred)
print(mae)

# MSE (Mean Squared Error)-
mse = mean_squared_error(y_true,y_pred)
print(mse)

# RMSE (Squared Root Of MSE)-
rmse = np.sqrt(mse)
print(rmse)

# r2 Score-
r2 = r2_score(y_true, y_pred)
print(r2)