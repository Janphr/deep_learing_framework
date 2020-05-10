//
// Created by janphr on 29.04.20.
//

#include "CrossEntropyLayer.h"

#include <utility>

void CrossEntropyLayer::forward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*L = -sum(t_i * log(x_i)*/
//    cout << "CrossEntropyLayer Forward:" << endl;

//    cout <<"Target: " << this->targets[0].getElements() << endl;
    double loss;
    for(int i = 0; i < in.size(); i++){
        MatrixXd tmp(in[i].getElements().rows(), in[i].getElements().cols());

        tmp.array() = this->targets[idx].getElements().array() * in[i].getElements().array().log();
        loss = -tmp.array().sum();
        tmp.resize(1,1);
        tmp.coeffRef(0) = loss;
        out[i].setElements(tmp);

    }

//    cout << "Loss: " << out[0].getElements() << endl;

}

void CrossEntropyLayer::backward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*dL/dX_i = -t_i/X_i*/
//    cout << "CrossEntropyLayer Backward:" << endl;

    for(int i = 0; i < in.size(); i++){
        MatrixXd tmp(out[i].getElements().rows(), out[i].getElements().cols());

        tmp.array() = - this->targets[idx].getElements().array() / out[i].getElements().array();

        out[i].setDeltas(tmp);
    }

//    cout << out[0].getDeltas() << endl;
}

CrossEntropyLayer::CrossEntropyLayer(vector<Tensor> targets) : targets(std::move(targets)) {}

void CrossEntropyLayer::update(SGDTrainer &trainer) {
    targetIndex++;
    if(targetIndex == this->targets.size())
        targetIndex = 0;
}




