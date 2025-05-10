import numpy as np
import pandas as pd
import math
import pickle
from typing import Dict, Any, Tuple

def mse(actual, predicted):
    """Mean squared error loss function"""
    return np.mean((actual-predicted)**2)


def mse_grad(actual, predicted):
    """Gradient of mean squared error"""
    return (predicted - actual)


def standard_scale(df, columns):
    """Standardize data by removing mean and scaling to unit variance"""
    scaled_df = df[columns].copy()
    for column in columns:
        mean = scaled_df[column].mean()
        std = scaled_df[column].std()
        scaled_df[column] = (scaled_df[column] - mean) / std
    return scaled_df


def init_params(layer_conf):
    """Initialize RNN layer parameters"""
    layers = []
    for i in range(1, len(layer_conf)):
        k = 1 / math.sqrt(layer_conf[i]["hidden"])
        
        i_weight = np.random.rand(layer_conf[i-1]["units"], layer_conf[i]["hidden"]) * 2 * k - k
        h_weight = np.random.rand(layer_conf[i]["hidden"], layer_conf[i]["hidden"]) * 2 * k - k
        h_bias = np.random.rand(1, layer_conf[i]["hidden"]) * 2 * k - k

        o_weight = np.random.rand(layer_conf[i]["hidden"], layer_conf[i]["output"]) * 2 * k - k
        o_bias = np.random.rand(1, layer_conf[i]["output"]) * 2 * k - k

        layers.append(
            [i_weight, h_weight, h_bias, o_weight, o_bias]
        )
    return layers


def forward(x, layers):
    """Forward pass through RNN layers"""
    outputs = []
    hiddens = []    
    for i in range(len(layers)):
        i_weight, h_weight, h_bias, o_weight, o_bias = layers[i] # Get all of the info
        hidden = np.zeros((x.shape[0], i_weight.shape[1])) # shape = (num inputs, num of hidden units)
        output = np.zeros((x.shape[0], o_weight.shape[1])) # shape = (num inputs, num of output units)

        for j in range(x.shape[0]): # Go through all of the inputs
            input_x = x[j,:][np.newaxis,:] @ i_weight # apply weights to x
            hidden_x = input_x + hidden[max(j-1,0),:][np.newaxis,:] @ h_weight + h_bias # gets current higgen state, apply weights, add biases and current input_x
            hidden_x = np.tanh(hidden_x) # activation function
            hidden[j,:] = hidden_x

            # output 
            output_x = hidden_x @ o_weight + o_bias
            output[j,:] = output_x

        hiddens.append(hidden)
        outputs.append(output)
    return hiddens, outputs[-1]


def backward(layers, x, lr, grad, hiddens):
    """Backward pass through RNN layers"""
    for i in range(len(layers)):
        i_weight, h_weight, h_bias, o_weight, o_bias = layers[i]  # Get layer parameters
        hidden = hiddens[i]  # Hidden states for current layer
        next_h_grad = None

        # Initialize gradients
        o_weight_grad = np.zeros_like(o_weight)
        o_bias_grad = np.zeros_like(o_bias)
        h_weight_grad = np.zeros_like(h_weight)
        h_bias_grad = np.zeros_like(h_bias)
        i_weight_grad = np.zeros_like(i_weight)

        for j in range(x.shape[0] - 1, -1, -1):  # Backprop through time
            out_grad = grad[j][np.newaxis, :]  # Shape (1, output_dim)

            # Output weight and bias gradient
            o_weight_grad += hidden[j][:, np.newaxis] @ out_grad
            o_bias_grad += out_grad

            # Propagate to hidden
            h_grad = out_grad @ o_weight.T

            if j < x.shape[0] - 1:
                # Backprop through next hidden state's gradient
                hh_grad = next_h_grad @ h_weight.T
                h_grad += hh_grad

            # Apply tanh derivative
            tanh_deriv = 1 - hidden[j][np.newaxis, :] ** 2
            h_grad = np.multiply(h_grad, tanh_deriv)

            next_h_grad = h_grad.copy()

            if j > 0:
                h_weight_grad += hidden[j - 1][:, np.newaxis] @ h_grad
                h_bias_grad += h_grad

            i_weight_grad += x[j][:, np.newaxis] @ h_grad

        # Normalize and apply gradients
        scale = lr / x.shape[0]
        i_weight -= i_weight_grad * scale
        h_weight -= h_weight_grad * scale
        h_bias -= h_bias_grad * scale
        o_weight -= o_weight_grad * scale
        o_bias -= o_bias_grad * scale

        layers[i] = [i_weight, h_weight, h_bias, o_weight, o_bias]

    return layers


def predict_next_candle(model, last_sequence, original_data, predictors, target):
    """
    Predict the next candle's closing price using the trained model
    
    Args:
        model: The trained RNN model (layers)
        last_sequence: The most recent sequence of candles
        original_data: The original dataframe with price statistics
        predictors: List of predictor column names
        target: Target column name
    
    Returns:
        float: Predicted next closing price
    """
    # Get the most recent sequence of data
    input_seq = last_sequence.copy()
    
    # Forward pass through the model
    _, prediction = forward(input_seq, model)
    
    # Get the last prediction (next time step)
    next_candle_normalized = prediction[-1][0]
    
    # Denormalize the prediction back to original price scale
    target_mean = original_data[target].mean()
    target_std = original_data[target].std()
    next_candle_price = next_candle_normalized * target_std + target_mean
    
    return next_candle_price


def train_model(train_x, train_y, valid_x, valid_y, layer_conf, epochs=15, lr=1e-5, sequence_len=100):
    """
    Train the RNN model on the provided data
    
    Args:
        train_x: Training features
        train_y: Training targets
        valid_x: Validation features
        valid_y: Validation targets
        layer_conf: Layer configuration
        epochs: Number of training epochs
        lr: Learning rate
        sequence_len: Sequence length for RNN
        
    Returns:
        The trained model layers
    """
    # Initialize model parameters
    layers = init_params(layer_conf)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for j in range(train_x.shape[0] - sequence_len):
            seq_x = train_x[j: (j + sequence_len),]
            seq_y = train_y[j: (j + sequence_len),]
            hiddens, outputs = forward(seq_x, layers)
            grad = mse_grad(seq_y, outputs)
            layers = backward(layers, seq_x, lr, grad, hiddens)
            epoch_loss += mse(seq_y, outputs)

        # Validation
        valid_loss = 0
        for j in range(valid_x.shape[0] - sequence_len):
            seq_x = valid_x[j: (j+sequence_len),]
            seq_y = valid_y[j: (j+sequence_len),]
            _, outputs = forward(seq_x, layers)
            valid_loss += mse(seq_y, outputs)
        
        if epoch % 10 == 0:  # Print less frequently to reduce output noise
            print(f"Epoch: {epoch} train loss {epoch_loss / len(train_x):.6f} valid loss {valid_loss / len(valid_x):.6f}")

    return layers


def save_model(layers, ticker):
    """Save model to disk"""
    model_path = f"model_{ticker}.npy"
    with open(model_path, 'wb') as f:
        pickle.dump(layers, f)
    print(f"Model saved to {model_path}")
    return model_path


def load_model(ticker):
    """Load model from disk"""
    model_path = f"model_{ticker}.npy"
    try:
        with open(model_path, 'rb') as f:
            layers = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return layers
    except FileNotFoundError:
        print(f"Model file {model_path} not found")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_signal_strength(magnitude: float, confidence: float) -> str:
    """Evaluate signal strength based on magnitude and confidence"""
    combined_strength = abs(magnitude) * confidence
    
    if combined_strength > 0.05:
        return "strong"
    elif combined_strength > 0.02:
        return "moderate"
    else:
        return "weak"