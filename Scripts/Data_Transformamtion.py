import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def clean_csv(df):
    """
    Elimina las columnas 'name', 'class', 'origin', 'skill_name'.
    Separa la columna 'skill_cost' en 'inicial_mana' y 'skill_cost'.
    Convierte las nuevas columnas a tipo numérico.
    Escala todas las columnas excepto 'cost' usando MinMaxScaler.
    Mueve la columna 'cost' al final del DataFrame.
    """
    # Eliminar columnas
    df = df.drop(['name', 'class', 'origin', 'skill_name'], axis=1)
    
    # Separar skill_cost en dos columnas
    skill_split = df['skill_cost'].str.split('/', expand=True)
    df['inicial_mana'] = skill_split[0]
    df['skill_cost'] = skill_split[1]
    df['inicial_mana'] = pd.to_numeric(df['inicial_mana'], errors='coerce').fillna(0)
    df['skill_cost'] = pd.to_numeric(df['skill_cost'], errors='coerce').fillna(0)

    pd.plotting.scatter_matrix(df) 
    plt.show()  
    
    # Escalar todas las columnas excepto 'cost'
    features = df.drop('cost', axis=1)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    
    # Unir con la columna 'cost'
    features_scaled_df['cost'] = df['cost'].values
    
    # Mover 'cost' al final (ya está al final)
    return features_scaled_df

df = pd.read_csv('./Datasets/TFT_set_10_cubic.csv')
df = clean_csv(df)

# Guardar el DataFrame transformado a un nuevo CSV
df.to_csv('./Datasets/TFT_set_10_cubic.csv', index=False)