import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def hypothesis(params, sample):
    """
    Calculates the hypothesis for a given sample and parameters using the MSE function.
    Args:
        params (list): The parameters for the MSE regression.
        sample (list): The input features for the sample.
    Returns:
        float: The hypothesis value between 0 and 1.
    """
    return params[0] + sum(p * s for p, s in zip(params[1:], sample))

def compute_error(params, samples, y):
    """
    Computes the mean error for the logistic regression model.
    Args:
        params (list): The parameters for the logistic regression.
        samples (list of lists): The input features for all samples.
        y (list): The true labels for all samples.
    Returns:
        float: The mean error.
    """
    total_error = 0
    for sample, label in zip(samples, y):
        pred = hypothesis(params, sample)
        total_error += (pred - label) ** 2
    return total_error / len(samples)

def gradient_descent(params, samples, y, alpha):
    """
    Performs one iteration of gradient descent to update the parameters.
    Args:
        params (list): The current parameters for the logistic regression.
        samples (list of lists): The input features for all samples.
        y (list): The true labels for all samples.
        alpha (float): The learning rate.
    Returns:
        list: The updated parameters after one iteration.
    """
    temp_params = params.copy()
    m = len(samples)
    gradient_bias = sum(hypothesis(params, sample) - label for sample, label in zip(samples, y))
    temp_params[0] = params[0] - alpha * (gradient_bias / m)
    for j in range(len(params)):
        gradient = sum((hypothesis(params, sample) - label) * sample[j-1] for sample, label in zip(samples, y))
        temp_params[j] = params[j] - alpha * (gradient / m)
    return temp_params

def round_off(num):
    """
    Redondea el número al entero más cercano.
    Si la parte decimal es 0.5 o mayor, redondea hacia arriba.
    Si es menor, redondea hacia abajo.
    """
    return int(num + 0.5)

def graph_errors(index):
    """
    Plots the error values over epochs.
    """
    plt.plot(__errors__[index], label='Entrenamiento')
    plt.plot(__val_errors__[index], label='Validación')
    plt.xlabel(f'Block {index} Epochs')
    plt.ylabel('Mean Error')
    plt.legend()
    plt.show()

def cross_validate(df, x=5, epochs=1000, alpha=0.05):
    """
    Realiza validación cruzada dividiendo el DataFrame en bloques de tamaño x.
    Entrena con los datos restantes y evalúa con el bloque separado.
    Retorna el error de entrenamiento de cada época en la variable errors.
    """

    n = len(df)
    errors = []
    global all_labels
    global all_preds
    global __val_errors__
    __val_errors__ = [] 
    for start in range(0, n, x):
        # Dividir en conjunto de validación y entrenamiento
        val_df = df.iloc[start:start+x]
        train_df = pd.concat([df.iloc[:start], df.iloc[start+x:]])

        # Preparar datos de entrenamiento
        train_features = train_df.drop('cost', axis=1)
        train_samples = train_features.values.tolist()
        train_labels = train_df['cost'].tolist()
        params = [1.0] * (train_features.shape[1] + 1)
        block_epoch_errors = []
        block_val_errors = []

        # Preparar datos de validación
        val_features = val_df.drop('cost', axis=1)
        val_samples = val_features.values.tolist()
        val_labels = val_df['cost'].tolist()

        for epoch in range(epochs):
            # Actualizar parámetros y calcular error de entrenamiento
            params = gradient_descent(params, train_samples, train_labels, alpha)
            epoch_error = compute_error(params, train_samples, train_labels)
            block_epoch_errors.append(epoch_error)

            # Calcular error de validación en cada época
            val_preds = [hypothesis(params, sample) for sample in val_samples]
            # val_preds = [round_off(p) for p in val_preds]
            val_epoch_error = np.mean([(p - y) ** 2 for p, y in zip(val_preds, val_labels)])
            block_val_errors.append(val_epoch_error)

        __errors__.append(block_epoch_errors)
        __val_errors__.append(block_val_errors)

        # Predicciones y evaluación
        preds = [hypothesis(params, sample) for sample in val_samples]
        preds = [round_off(p) for p in preds]
        all_labels = all_labels + val_labels if 'all_labels' in globals() else val_labels
        all_preds = all_preds + preds if 'all_preds' in globals() else preds
        print(f'Predictions: {preds}')
        print(f'Actual: {val_labels}')
        errors = np.mean([(p - y) ** 2 for p, y in zip(preds, val_labels)])
        print(f'Block {start//x}: Error = {errors:.4f}')
        __block_errors__.append(errors)

    print(f'\nAverage validation error: {np.mean(__block_errors__):.4f}')
    return errors

def confusion_matrix_costs(y_true, y_pred, labels=[1,2,3,4,5,6]):
    """
    Muestra la matriz de confusión para los costos predichos vs reales.
    Args:
        y_true (list): Lista de costos reales.
        y_pred (list): Lista de costos predichos.
        labels (list): Lista de posibles valores de costo.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # print("Matriz de confusión (filas: real, columnas: predicho):")
    # print(pd.DataFrame(cm, index=[f"Real {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión de Costos')
    plt.show()

global __errors__
__errors__ = []
global __block_errors__
__block_errors__ = []
df = pd.read_csv('./TFT_Champion_Transformed.csv')
df = df.sample(frac=1).reset_index(drop=True)
block_errors = cross_validate(df, x=5, epochs=1000, alpha=0.1)
confusion_matrix_costs(all_labels, all_preds)
for i in range(len(__errors__)):
    graph_errors(i)

