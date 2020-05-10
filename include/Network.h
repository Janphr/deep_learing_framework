//
// Created by janphr on 29.04.20.
//

#pragma once

#include "InputLayer.h"
#include "SGDTrainer.h"

class Network {

public:
    explicit Network(vector<shared_ptr<Layer>>& layers, vector<vector<vector<Tensor>>>& tensors);
    void train(SGDTrainer &trainer);
    void run();
    void set_data(vector<vector<Tensor>> &data);
    void set_targets(vector<Tensor> &targets);
    void print_result(int every_x_dataset);
    void print_debug(int every_x_dataset);
    void reset_tensors(vector<vector<vector<Tensor>>> &tensors);
    void reset_deltas(int i);


private:
    vector<shared_ptr<Layer>> layers;
    vector<vector<vector<Tensor>>> tensors;
    vector<Tensor> targets;
//    vector<Tensor> parameters;
//    vector<Tensor> deltaParameters;

//    backprop(data)
//    forward()

};
