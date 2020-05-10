//
// Created by janphr on 29.04.20.
//

#include "SoftmaxLayer.h"
#include <unsupported/Eigen/AutoDiff>

void SoftmaxLayer::forward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*Y = softmax(X) = e^x_i/sum(e^x_j) | cancel e^MAX */
//    cout << "SoftmaxLayer Forward:" << endl;

    double sum;
    for(int i = 0; i < in.size(); i ++){
        MatrixXd tmp(in[i].getElements().rows(), in[i].getElements().cols());

        tmp.array() = in[i].getElements().array().exp();
        sum = tmp.array().sum();
        tmp.array() = tmp.array() / sum;

        out[i].setElements(tmp);
    }
//    cout << in[0].getElements() << endl;
//    cout << out[0].getElements() << endl;
//    cout << in[5].getElements() << endl;
//    cout << out[5].getElements() << endl;
}

void SoftmaxLayer::backward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*dX = dY * jacobi(dX/dY)*/
//    cout << "SoftmaxLayer Backward:" << endl;

    for(int i = 0; i < in.size(); i++){
        MatrixXd J(in[i].getElements().cols(), in[i].getElements().cols());

        for(int k = 0; k < in[i].getElements().cols(); k++){
            for(int j = 0; j < in[i].getElements().cols(); j++){
                if(k == j){
                    J.coeffRef(k, j) = in[i].getElements().coeffRef(k) * (1 - in[i].getElements().coeffRef(k));
                } else {
                    J.coeffRef(k, j) = -in[i].getElements().coeffRef(k) * in[i].getElements().coeffRef(j);
                }
            }
        }

        out[i].setDeltas(in[i].getDeltas() * J);
    }
//    cout << in[0].getDeltas() << endl;
//    cout << out[0].getDeltas() << endl;
//    cout << in[5].getDeltas() << endl;
//    cout << out[5].getDeltas() << endl;
}
