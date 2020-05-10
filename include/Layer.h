//
// Created by janphr on 29.04.20.
//

#pragma once

#include <iostream>
#include "Tensor.h"
#include "SGDTrainer.h"

using namespace std;

class Layer {

public:
    virtual void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) = 0;
    virtual void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) {};
    virtual void update(SGDTrainer &trainer){};

};


