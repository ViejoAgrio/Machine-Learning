import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def hypothesis(params, sample):
    """
    Calculates the hypothesis for a given sample and parameters using the MSE function.
    Args:
        params (list): The parameters for the MSE regression.
        sample (list): The input features for the sample.
    Returns:
        float: The hypothesis value between 0 and 1.
    """
    return sum(p * s for p, s in zip(params, sample))

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
    for j in range(len(params)):
        gradient = sum((hypothesis(params, sample) - label) * sample[j] for sample, label in zip(samples, y))
        temp_params[j] = params[j] - alpha * (gradient / len(samples))
    return temp_params

def predict_champion_cost(params, champ, feature_names, scaler):
    """
    Predicts the cost of a new champion based on its features.
    Args:
        params (list): The learned parameters from the model.
        champ (list): The features of the new champion.
        feature_names (list): The names of the features used in the model.
        scaler (MinMaxScaler): The scaler used to normalize the features.
    Returns:
        float: The predicted cost of the champion.
    """
    champ_df = pd.DataFrame([champ], columns=feature_names)
    champ_scaled = scaler.transform(champ_df)[0]
    return hypothesis(params, champ_scaled)

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
    plt.plot(__errors__[index])
    plt.xlabel(f'Block {index} Epochs')
    plt.ylabel('Mean Error')
    plt.show()

def cross_validate(df, x=5, epochs=1000, alpha=0.05):
    """
    Realiza validación cruzada dividiendo el DataFrame en bloques de tamaño x.
    Entrena con los datos restantes y evalúa con el bloque separado.
    Guarda el error de entrenamiento de cada época en la variable global __errors__.
    """
    global __errors__
    n = len(df)
    errors = []
    feature_names = df.drop('cost', axis=1).columns.tolist()

    for start in range(0, n, x):
        val_df = df.iloc[start:start+x]
        train_df = pd.concat([df.iloc[:start], df.iloc[start+x:]])

        train_features = train_df.drop('cost', axis=1)
        train_samples = train_features.values.tolist()
        train_labels = train_df['cost'].tolist()

        params = [1.0] * train_features.shape[1]

        block_epoch_errors = []

        for epoch in range(epochs):
            params = gradient_descent(params, train_samples, train_labels, alpha)
            epoch_error = compute_error(params, train_samples, train_labels)
            block_epoch_errors.append(epoch_error)

        __errors__.append(block_epoch_errors)

        val_features = val_df.drop('cost', axis=1)
        val_samples = val_features.values.tolist()
        val_labels = val_df['cost'].tolist()

        preds = [hypothesis(params, sample) for sample in val_samples]
        preds = [round_off(p) for p in preds]
        print(f'Predictions: {preds}')
        print(f'Actual: {val_labels}')
        errors = np.mean([(p - y) ** 2 for p, y in zip(preds, val_labels)])
        print(f'Block {start//x}: Error = {errors:.4f}')

    print(f'\nAverage validation error: {np.mean(errors):.4f}')
    return errors

global __errors__
__errors__ = []
df = pd.read_csv('./TFT_Champion_Transformed.csv')
df = df.sample(frac=1).reset_index(drop=True)
block_errors = cross_validate(df, x=5, epochs=1000, alpha=0.1)
for i in range(len(__errors__)):
    graph_errors(i)

