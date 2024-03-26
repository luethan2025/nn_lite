/**
 * @file linear.cpp
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Linear object implementation
 * @date 2024-03-24
 * 
 * @copyright Copyright (c) 2024
 */
#include "linear.h"

/**
 * @brief Constructor for the Linear object
 *
 * @param input_size Input dimension of the layer
 * @param output_size Output dimension of the layer
 * @param lr Learning rate
 * @param initialization Name of an initialization method
 *
 * @throws runtime_error if the input dimension is not greater than 0
 * @throws runtime_error if the output dimension is not greater than 0
 * @throws runtime_error if the learning rate is negative
 * @throws runtime_error if the requested initialization method is not "Xavier"
 *         or "He"
 */
Linear::Linear(int input_size, int output_size, double eta, string initialization) {
  // assert that the input dimension is greater than 0
  if (input_size > 0) {
    in_dim = input_size;
  } else {
    throw runtime_error("Input size must be positive and non-zero");
  }
  
  // assert that the output dimension is greater than 0
  if (output_size > 0) {
    out_dim = output_size;
  } else {
    throw runtime_error("Output size must be positive and non-zero");
  }

  // assert that the learning rate is at least 0
  if (eta >= 0) {
    lr = eta;
  } else {
    throw runtime_error("The learning rate must be at least zero");
  }
  
  // define a distribution based on the initialization method
  normal_distribution<double> distribution;
  if (initialization == "Xavier") {
    // create a random number distribution that produces double-precision
    // floating point according to the normal distribution for Uniform Xavier
    // initialization
    distribution = normal_distribution<double>(0.0, sqrt(6.0 / (in_dim + out_dim)));
  } else if (initialization == "He") {
    // create a random number distribution that produces double-precision
    // floating point according to the normal distribution for He
    // initialization
    distribution = normal_distribution<double>(0.0, 2.0 / in_dim);
  } else {
    throw runtime_error("Iitialization method has not implemented");
  }

  default_random_engine generator;

  // initialize the weight matrix
  for (int o = 0; o < out_dim; o++) {
    weights.push_back(vector<double>());
    for (int i = 0; i < in_dim; i++) {
      weights.back().push_back(distribution(generator));
    }
  }

  // create a zero vector for the bias vector
  for (int o = 0; o < output_size; o++) {
    bias.push_back(0.0);
  }
}

/**
 * @brief Forward propogation through Linear
 *
 * @note 
 * The linear activation can be described by the function:
 *   y = x * A.T + b
 *
 * @note 
 * The value of the vector in_vals is set in this function
 *
 * @param x Input vector for this activation function
 * Should have dimensions (1, in_dim)
 * 
 * @return Output vector of this activation function
 * Should have dimensions (1, out_dim)
 *
 * @throws runtime_error if the length of the input vector does not match the
 *         length of the layer's input dimension
 */
vector<double> Linear::forward(const vector<double> &x) {
  // assert than the length of the input vector is equal to the length of the
  // layer's input dimension
  if (x.size() != in_dim) {
    throw runtime_error("Dimension mismatch");
  }

  // save a copy of the input vector
  in_vals = x;

  vector<double> y;
  // apply the activation function to the input vector
  for (int o = 0; o < out_dim; o++) {
    double res = 0.0;
    for (int i = 0; i < in_dim; i++) {
      res += (weights[o][i] * x[i]);
    }
    // account for the bias term
    res += bias[o];

    y.push_back(res);
  }

  return y;
}

/**
 * @brief Backward propogation for Linear
 *
 * @note
 * The weight matrix and the bias vector are updated using the Mean Square
 * Error (MSE)/L2 loss function:
 *   E = (y - ŷ)^2
 * 
 * @param grad Gradient (Loss w.r.t. data) flowing backwards from the next
 *             layer
 * Should have dimensions (1, out_dim)
 *  
 * @return Gradient (Loss w.r.t. data) for the inputs of this layer
 * Should have dimensions (1, in_dim)
 *
 * @throws runtime_error if the length of the loss from the previous layer does
 *         not match the length of the layer's output dimension
 */
vector<double> Linear::backward(const vector<double> &grad) {
  // assert that the length of the loss from the previous layer is equal to the
  // length of the layer's output dimension
  if (grad.size() != out_dim) {
    throw runtime_error("Dimension mismatch");
  }

  vector<double> prev_grad;

  // compute the weighted sum of the loss for each input dimension
  for (int i = 0; i < in_dim; i++) {
    double g = 0.0;
    for (int o = 0; o < out_dim; o++) {
      g += grad[o] * weights[o][i];
    }
    prev_grad.push_back(g);
  }

  // update the weight matrix and bias vector accordingly
  for (int o = 0; o < out_dim; o++) {
    for (int i = 0; i < in_dim; i++) {
      weights[o][i] -= lr * grad[o] * in_vals[i];
    }
    bias[o] -= lr * grad[o];
  }

  return prev_grad;
}
