#include <iostream>
#include "Linear.h"

using namespace std;

int main() {
  cout << "--------------------------------------------------------------------"
       << endl;

  cout << "Creating a linear layer with in_dim of size 4 and out_dim of size 1"
       << endl;
  cout << "Initializing weight matrix using Xavier initialization"
       << endl
       << endl;
  Linear L(4, 1, "Xavier");

  vector<vector<double>> weights = L.get_weights();
  vector<double> bias = L.get_bias();

  cout << "Testing forward pass using vector {1.0, 1.0, 1.0, 1.0}..."
       << endl;

  vector<double> x {1.0, 1.0, 1.0, 1.0};
  vector<double> y = L.forward(x);

  try {
    if ((weights[0][0] + bias[0]) + (weights[0][1] + bias[0]) +
        (weights[0][2] + bias[0]) + (weights[0][3] + bias[0]) == y[0]) {
      cout << "Forward pass returned the correct value"
           << endl
           << endl;
    } else {
      throw runtime_error("Forward pass failed on vector {1.0, 1.0, 1.0, 1.0}");
    }
  } catch (const runtime_error e) {
    cout << e.what()
         << endl;
    cout << "Terminating unit tests"
         << endl;
    return -1;
  }

  cout << "Testing forward pass using vector {2.0, 0.0, 2.0, 0.0}..." << endl;

  x = {2.0, 0.0, 2.0, 0.0};
  y = L.forward(x);

  try {
    if ((2.0 * weights[0][0] + bias[0]) + (2.0 * weights[0][2] + bias[0]) == y[0]) {
      cout << "Forward pass returned the correct value"
           << endl
           << endl;
    } else {
      throw runtime_error("Forward pass failed on vector {2.0, 0.0, 2.0, 0.0}");
    }
  } catch (const runtime_error e) {
    cout << e.what()
         << endl;
    cout << "Terminating test"
         << endl;
    return -1;
  }

  cout << "Testing forward pass using vector {2.0, 1.0, 2.0, 1.0}..." << endl;

  x = {2.0, 1.0, 2.0, 1.0};
  y = L.forward(x);

  try {
    if ((2.0 * weights[0][0] + bias[0]) + (weights[0][1] + bias[0]) +
        (2.0 * weights[0][2] + bias[0]) + (weights[0][3] + bias[0]) == y[0]) {
      cout << "Forward pass returned the correct value"
           << endl; 
    } else {
      throw runtime_error("Forward pass failed on vector {2.0, 1.0, 2.0, 1.0}");
    }
  } catch (const runtime_error e) {
    cout << e.what()
         << endl;
    cout << "Terminating test"
         << endl;
    return -1;
  }

  cout << "--------------------------------------------------------------------"
       << endl
       << endl;

  cout << "--------------------------------------------------------------------"
       << endl;

  cout << "Creating a linear layer with in_dim of size 3 and out_dim of size 2"
       << endl;
  cout << "Initializing weight matrix using He initialization"
       << endl
       << endl;
  L = Linear(3, 2, "He");

  weights = L.get_weights();
  bias = L.get_bias();

  cout << "Testing forward pass using vector {2.0, -1.0, 0.0}..."
       << endl;
  
  x = {2.0, -1.0, 0.0};
  y = L.forward(x);

  try {
    if ((2.0 * weights[0][0] + bias[0]) + (-1.0 * weights[0][1] + bias[0]) == y[0] &&
        (2.0 * weights[1][0] + bias[0]) + (-1.0 * weights[1][1] + bias[0]) == y[1]) {
      cout << "Forward pass returned the correct value"
           << endl
           << endl;
    } else {
      throw runtime_error("Forward pass failed on vector {2.0, -1.0, 0.0}");
    }
  } catch (const runtime_error e) {
    cout << e.what() << endl;
    cout << "Terminating test" << endl;
    return -1;
  }

  cout << "Testing forward pass using vector {0.0, -1.0, 0.0}..." << endl;
  
  x = {0.0, -1.0, 0.0};
  y = L.forward(x);

  try {
    if ((-1.0 * weights[0][1] + bias[0]) == y[0] &&
        (-1.0 * weights[1][1] + bias[0]) == y[1]) {
      cout << "Forward pass returned the correct value"
           << endl;
    } else {
      throw runtime_error("Forward pass failed on vector {0.0, -1.0, 0.0}");
    }
  } catch (const runtime_error e) {
    cout << e.what()
         << endl;
    cout << "Terminating test"
         << endl;
    return -1;
  }

  cout << "--------------------------------------------------------------------"
       << endl
       << endl;

  cout << "Linear layer returns the correct forward pass values with no errors"
       << endl; 

  return 0;
}
