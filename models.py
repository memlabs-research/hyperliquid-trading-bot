"""
Machine Learning Models Module

This module provides lightweight implementations of machine learning models
for making predictions in production environments. The models are designed
to be used with pre-trained weights (trained offline) and focus on fast
inference for real-time applications like algorithmic trading.

Note: This module provides inference only. Model training should be done
separately using frameworks like scikit-learn, PyTorch, or TensorFlow.
"""

import numpy as np


class LinReg:
    """
    Linear Regression model for prediction tasks.

    This class implements a simple linear regression model that performs
    predictions using the formula: y = X·w + b, where:
    - X is the input feature vector
    - w is the weight vector
    - b is the bias (intercept) term

    Linear regression is commonly used in quant trading.

    The model assumes weights and bias have been pre-trained externally
    and are loaded for inference only.

    Example:
        >>> # Suppose we trained a model to predict stock returns
        >>> # based on [log_return, volume, volatility]
        >>> weights = np.array([0.5, 0.2, -0.3])
        >>> bias = 0.01
        >>>
        >>> model = LinReg(weights=weights, bias=bias)
        >>>
        >>> # Make a prediction with new features
        >>> features = np.array([0.02, 1500000, 0.15])
        >>> prediction = model.predict(features)
        >>> print(f"Predicted return: {prediction:.4f}")

    Example with multiple predictions:
        >>> # Predict on a batch of samples
        >>> features_batch = np.array([
        ...     [0.02, 1500000, 0.15],
        ...     [0.01, 1200000, 0.12],
        ...     [-0.01, 1800000, 0.18]
        ... ])
        >>> predictions = model.predict(features_batch)
        >>> # Returns array of predictions, one per row
    """

    def __init__(self, weights: np.ndarray, bias: float):
        """
        Initialize the linear regression model with pre-trained parameters.

        Args:
            weights: Coefficient weights for each input feature.
                     Should be a 1D numpy array of shape (n_features,)
            bias: The intercept term (scalar value)

        Example:
            >>> # For a model with 3 features
            >>> weights = np.array([1.5, -0.8, 0.3])
            >>> bias = 0.5
            >>> model = LinReg(weights, bias)
        """
        self.weights = weights
        self.bias = bias

    def predict(self, x: np.ndarray) -> float:
        """
        Generate a prediction for the given input features.

        Computes the dot product of input features with weights,
        then adds the bias term. This implements the standard linear
        regression equation: y = X·w + b

        Args:
            x: Input feature vector or matrix.
               - For single prediction: 1D array of shape (n_features,)
               - For batch prediction: 2D array of shape (n_samples, n_features)

        Returns:
            Predicted value(s):
            - Single float if input is 1D (single sample)
            - 1D array if input is 2D (batch of samples)

        Example (single prediction):
            >>> model = LinReg(np.array([2.0, 3.0]), bias=1.0)
            >>> x = np.array([1.0, 2.0])
            >>> prediction = model.predict(x)
            >>> # Returns: 2.0*1.0 + 3.0*2.0 + 1.0 = 9.0

        Example (batch prediction):
            >>> x_batch = np.array([
            ...     [1.0, 2.0],
            ...     [2.0, 1.0]
            ... ])
            >>> predictions = model.predict(x_batch)
            >>> # Returns: [9.0, 8.0]

        Note:
            The input dimension must match the weight vector size.
            For a model with 5 features, x must have 5 elements (or
            n_samples × 5 for batches).
        """
        # Compute weighted sum of features plus bias
        # np.dot handles both 1D (single sample) and 2D (batch) inputs
        return np.dot(x, self.weights) + self.bias