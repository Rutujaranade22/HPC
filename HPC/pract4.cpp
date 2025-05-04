#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <climits>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid for backpropagation
double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// Function for computing dot product of two vectors
double dotProduct(const vector<double>& a, const vector<double>& b) {
    double result = 0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Forward pass to compute the output of the network
vector<double> feedforward(const vector<double>& inputs, const vector<vector<double>>& weights) {
    vector<double> outputs(weights.size());
    #pragma omp parallel for
    for (size_t i = 0; i < weights.size(); i++) {
        outputs[i] = sigmoid(dotProduct(inputs, weights[i]));
    }
    return outputs;
}

// Backpropagation to compute gradients and update weights
void backpropagate(vector<vector<double>>& weights, const vector<double>& inputs, const vector<double>& outputErrors, double learningRate) {
    #pragma omp parallel for
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < inputs.size(); j++) {
            weights[i][j] -= learningRate * outputErrors[i] * sigmoidDerivative(weights[i][j]);
        }
    }
}

int main() {
    // Set up a simple 2-layer neural network (1 input layer, 1 output layer)
    vector<double> inputs = {0.5, 0.6}; // Example inputs
    vector<vector<double>> weights = {
        {0.3, 0.5}, // Weights for first neuron
        {0.8, 0.9}  // Weights for second neuron
    };

    vector<double> targetOutputs = {1.0, 0.0}; // Desired outputs for training

    int epochs = 10000; // Number of training epochs
    double learningRate = 0.1;

    // Measure the start time
    auto start = high_resolution_clock::now();

    // Training the neural network using parallelized feedforward and backpropagation
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Perform feedforward
        vector<double> outputs = feedforward(inputs, weights);

        // Calculate output error (difference between predicted and target outputs)
        vector<double> outputErrors(outputs.size());
        for (size_t i = 0; i < outputs.size(); i++) {
            outputErrors[i] = targetOutputs[i] - outputs[i];
        }

        // Perform backpropagation and weight update
        backpropagate(weights, inputs, outputErrors, learningRate);

        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << " - Output: ";
            for (double output : outputs) {
                cout << output << " ";
            }
            cout << endl;
        }
    }

    // Measure the end time and output the time taken
    auto stop = high_resolution_clock::now();
    cout << "Training completed in: " << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;

    return 0;
}
