//
// Created by janphr on 29.04.20.
//

#pragma once

#include "Layer.h"

class ActivationLayer : public Layer  {

public:
    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

};

