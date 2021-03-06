//
// Created by janphr on 29.04.20.
//

#pragma once

#include "Layer.h"

class FullyConnectedLayer : public Layer {

public:
    FullyConnectedLayer(const Shape &inShape, const Shape &outShape);

    FullyConnectedLayer(Tensor weightMatrix, Tensor bias, const Shape &inShape, const Shape &outShape,
                        bool firstLayer);

    /*Y = X * W + bias*/
    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    /*dX = dY * W^T*/
    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    /*dL/dW = X^T * dY; dL/dbias = dY*/
    void update(SGDTrainer &trainer) override;

    Tensor weightMatrix;
    Tensor bias;

private:

    Shape inShape;
    Shape outShape;
    bool first_layer;

    static void trans(Tensor &tensor, Eigen::MatrixXd &transposed, const Shape &outShape);

};

