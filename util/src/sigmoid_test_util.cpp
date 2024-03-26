/**
 * @file sigmoid_test_util.h
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Sigmoid testing functions
 * @date 2024-03-26
 * 
 * @copyright Copyright (c) 2024
 */
#include "sigmoid_test_util.h"

/**
 * @brief Tests the forward pass of a Sigmoid object
 * 
 * @param layer Sigmoid object
 * @param x Input vector
 *
 * @throws runtime_error if forward pass does not return the correct value
 */
void test_forward_pass(Sigmoid layer, const vector<double> &x) {
  print_forward_pass_message(x);

  vector<double> y = layer.forward(x);
  try {
    for (int i = 0; i < x.size(); i++) {
      if (x[i] != y[i]) {
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
