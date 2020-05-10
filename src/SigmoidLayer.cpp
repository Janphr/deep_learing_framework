//
// Created by janphr on 29.04.20.
//

#include "SigmoidLayer.h"

void SigmoidLayer::forward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*Y = sigma(X) = 1/(1+e^-X) | element wise*/
//    cout << "SigmoidLayer Forward:" << endl;

    for(int i = 0; i < in.size(); i++){
        MatrixXd tmp(in[i].getElements().rows(), in[i].getElements().cols());

        tmp.array() = 1 / (1 + (-in[i].getElements().array()).exp());
        out[i].setElements(tmp);
    }
//    cout << in[0].getElements() << endl;
//    cout << out[0].getElements() << endl;
//    cout << in[5].getElements() << endl;
//    cout << out[5].getElements() << endl;
}

void SigmoidLayer::backward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*dX = [sima(X) * (1 - sigma(X)] (multiplied elementwise(hadama product?)) dY*/
//    cout << "SigmoidLayer Backward:" << endl;

    for(int i = 0; i < in.size(); i++){
        MatrixXd tmp(in[i].getDeltas().rows(), in[i].getDeltas().cols());

        tmp.array() = (in[i].getElements().array() * (1 - in[i].getElements().array())) * in[i].getDeltas().array();

        out[i].setDeltas(tmp);
    }
//    cout << in[0].getDeltas() << endl;
//    cout << out[0].getDeltas() << endl;
//    cout << in[5].getDeltas() << endl;
//    cout << out[5].getDeltas() << endl;
}

SigmoidLayer::SigmoidLayer() = default;
