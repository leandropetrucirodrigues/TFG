
import pandas as pd
olhos = pd.read_csv('olhos.csv')
olhos_AWS = pd.read_csv ('tabela_medicao_aws.csv')


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, median_absolute_error

y_true = olhos[['real_left_eye_x',	'real_left_eye_y',	'real_right_eye_x',	'real_right_eye_y']].values
y_pred = olhos[['pred_left_eye_x',	'pred_left_eye_y',	'pred_right_eye_x', 'pred_right_eye_y']].values

print('dlib')
print('MAE =',mean_absolute_error(y_true, y_pred))
print('MSE =',mean_squared_error(y_true, y_pred))
print('MAPE =',mean_absolute_percentage_error(y_true, y_pred))
print('R² =',r2_score(y_true, y_pred))
print('MedAE =',median_absolute_error(y_true, y_pred))

y_true_aws = olhos[['real_left_eye_x',	'real_left_eye_y',	'real_right_eye_x',	'real_right_eye_y']].values
y_pred_aws = olhos_AWS[['left_eye_X',	'left_eye_Y',	'right_eye_X', 
'right_eye_Y']].values


print('-----------------------------------')
print('AWS')
print('MAE =',mean_absolute_error(y_true_aws, y_pred_aws))
print('MSE =',mean_squared_error(y_true_aws, y_pred_aws))
print('MAPE =',mean_absolute_percentage_error(y_true_aws, y_pred_aws))
print('R² =',r2_score(y_true_aws, y_pred_aws))
print('MedAE =',median_absolute_error(y_true_aws, y_pred_aws))
