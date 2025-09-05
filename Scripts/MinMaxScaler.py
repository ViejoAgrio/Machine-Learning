from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('./Datasets/TFT_set_14_y_15.csv')
df = df.sample(frac=1).reset_index(drop=True)
features = df.drop('cost', axis=1)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
features_scaled_df['cost'] = df['cost'].values
features_scaled_df.to_csv('./Datasets/TFT_set_14_y_15_Scaled.csv', index=False)