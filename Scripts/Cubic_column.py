import pandas as pd
import numpy as np

def add_power3_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Crea una nueva columna con el mismo nombre + '^3'
    que contiene los valores de la columna original elevados al cubo.
    """
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame")
    
    df[column + "_tanh"] = np.tanh(df[column])
    # df = df.drop([column], axis=1)
    return df


if __name__ == "__main__":
    # Leer el CSV
    df = pd.read_csv("./Datasets/TFT_set_10_cubic.csv")  # Cambia por tu archivo

    # Ejemplo: elevar la columna HP al cubo
    df = add_power3_column(df, "attack")

    # Guardar el resultado
    df.to_csv("./Datasets/TFT_set_10_cubic.csv", index=False)
