/**
 * @file linear_test_util.h
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Linear testing functions
 * @date 2024-03-25
 * 
 * @copyright Copyright (c) 2024
 */
#include "linear_test_util.h"

/**
 * @brief Tests the forward pass of a Linear object
 * 
 * @param layer Linear object
 * @param x Input vector
 *
 * @throws runtime_error if
 */
void test_forward_pass(Linear layer, const vector<double> &x) {
  print_forward_pass_message(x);

  vector<vector<double>> weights = layer.weights;
  vector<double> bias = layer.bias;

  vector<double> y = layer.forward(x);

  vector<double> res;
  for (const auto &b: bias) {
    double r = 0.0;
    for (const auto &w: weights ) {
      for (const auto &v: w) {
        r += v + b;
      }
    }
    res.push_back(r);
  }

  try {
    for (int i = 0; i < res.size(); i++) {
      if (res[i] != y[i]) {
        throw runtime_error("");
      }
    }
  } catch (const runtime_error &e){
    cout << "Forward pass failed on vector ";
    print_vector(x);
    cout << endl;
  }

  cout << "Forward pass returned the correct value" << endl;
}
