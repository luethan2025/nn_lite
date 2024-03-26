/**
 * @file sigmoid.cpp
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Sigmoid object implementation
 * @date 2024-03-26
 * 
 * @copyright Copyright (c) 2024
 */
#include "sigmoid.h"

/**
 * @brief Forward propogation through Sigmoid
 *
 * @note
 * The sigmoid activiation can be described by the function:
 *   y = 1 / (1 + e^x)
 *
 * @note 
 * The value of the vector out_vals is set in this function
 * 
 * @param x Input vector for this activation function
 * 
 * @return Output vector of this activation function
 */
vector<double> Sigmoid::forward(const vector<double> &x) {
  out_vals = vector<double>();
  for(const auto &i: x) {
    out_vals.push_back(1.0 / (1.0 + exp(-i)));
  }
  return out_vals;
}

/**
 * @brief Backward propogation for Sigmoid
 *
 * @note
 * The derivative of the sigmoid function is:
 *   y(x) * (1 - y(x)) where y(x) if the sigmoid function 
 * 
 * @param grad Gradient (Loss w.r.t. data) flowing backwards from the next
 *             layer
 * Should have dimensions (1, out_val.size())
 *
 * @return Gradient (Loss w.r.t. data) for the inputs of this layer
 *
 * @throws runtime_error if the length of the loss from the previous layer does
 *         not match the length of the layer's output vector (from forward
 *         pass)
 */
vector<double> Sigmoid::backward(vector<double> grad) {
  // assert that the length of the loss from the previous layer is equal to the
  // length of the layer's output vector (from forward pass)
  if (grad.size() != out_vals.size()) {
    throw runtime_error("Dimension mismatch");
  }

  for (int o = 0; o < out_vals.size(); o++) {
    grad[o] *= out_vals[o] * (1.0 - out_vals[o]);
  }

  return grad;
}
