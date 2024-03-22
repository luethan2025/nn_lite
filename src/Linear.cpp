#include <random>
#include <cmath>
#include "Linear.h"

using namespace std;

Linear::Linear(int input_size, int output_size, string initialization) {
  if (input_size > 0) {
    in_dim = input_size;
  } else {
    throw logic_error("Input size must be positive and non-zero");
  }
  
  if (output_size > 0) {
    out_dim = output_size;
  } else {
    throw logic_error("Output size must be positive and non-zero");
  }
  

  default_random_engine generator;
  normal_distribution<double> distribution;
  if (initialization == "Xavier") {
    normal_distribution<double> distribution(0.0, sqrt(6.0 / (input_size + output_size)));
  } else if (initialization == "He") {
    normal_distribution<double> distribution(0.0, 2.0 / input_size);
  } else {
    throw logic_error("Weight initialization method has not implemented");
  }

  for (int o = 0; o < output_size; o++) {
    weights.push_back(vector<double>());
    for (int i = 0; i < input_size; i++) {
      weights.back().push_back(distribution(generator));
    }
  }

  for (int o = 0; o < output_size; o++) {
    bias.push_back(0.0);
  }
}

vector<vector<double>> Linear::get_weights() {
  vector<vector<double>> weights_copy = vector<vector<double>>();
  for (int o = 0; o < out_dim; o++) {
    weights_copy.push_back(vector<double>());
    for (int i = 0; i < in_dim; i++) {
      weights_copy.back().push_back(weights[o][i]);
    }
  }
  return weights_copy;
}

vector<double> Linear::get_bias() {
  vector<double> bias_copy = vector<double>();
  for (int o = 0; o < out_dim; o++) {
    bias_copy.push_back(bias[o]);
  }
  return bias_copy;
}

vector<double> Linear::forward(vector<double> x) {
  if (x.size() != in_dim) {
    throw logic_error("Dimension mismatch");
  }

  vector<double> y = vector<double>();

  for (int o = 0; o < out_dim; o++) {
    double result = 0.0;
    for (int w = 0; w < in_dim; w++) {
      result += weights[o][w] * x[w];
    }
    result += bias[o];
    y.push_back(result);
  }
  return y;
}
