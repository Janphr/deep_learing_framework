//
// Created by janphr on 29.04.20.
//

#include "FullyConnectedLayer.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include <utility>


void FullyConnectedLayer::forward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*Y = X * W + bias*/
//    cout << "FullyConnectedLayer Forward:" << endl;

    for(int i = 0; i < in.size(); i++){
        Eigen::MatrixXd tmp(this->inShape.r, this->inShape.c);

        tmp.noalias() =  in[i].getElements() * this->weightMatrix.getElements();
        tmp.array() += this->bias.getElements().array();

        tmp.resize(this->outShape.r, this->outShape.c);

        out[i].setElements(tmp);
    }

//    cout << in[0].getElements() << endl;
//    cout << out[0].getElements() << endl;
//    cout << in[5].getElements() << endl;
//    cout << out[5].getElements() << endl;
}

void FullyConnectedLayer::backward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*dX = dY * W^T*/

//    Eigen::MatrixXd transposed;
    if(!first_layer) {
        for (int i = 0; i < in.size(); i++) {
//            trans(this->weightMatrix, transposed,
//                  Shape(this->weightMatrix.getElements().rows(), this->weightMatrix.getElements().cols(), 0));
            out[i].setDeltas(in[i].getDeltas() * this->weightMatrix.getElements().transpose());
            out[i].getDeltas().resize(this->inShape.r, this->inShape.c);
        }
    }
    /*dL/dW = X^T * dY; dL/dbias = dY*/
    for(int i = 0; i < in.size(); i++){
//        trans(out[i], transposed, this->outShape);
            this->weightMatrix.getDeltas().noalias() = out[i].getElements().transpose() * in[i].getDeltas();
            this->bias.getDeltas() = in[i].getDeltas();
    }


//    update(in, out);
//    cout << "FullyConnectedLayer Backward:" << endl;
//    cout << in[0].getDeltas() << endl;
//    cout << out[0].getDeltas() << endl;
//    cout << in[5].getDeltas() << endl;
//    cout << out[5].getDeltas() << endl;
}

void FullyConnectedLayer::update(SGDTrainer &trainer) {
    /*dL/dW = X^T * dY; dL/dbias = dY*/

//    Eigen::Matrix<double, Eigen::Dynamic, 1>::ConstAlignedMapType dw(this->weightMatrix.getDeltas().data(), this->weightMatrix.getDeltas().size());
//    Eigen::Matrix<double, Eigen::Dynamic, 1>::ConstAlignedMapType db(this->bias.getDeltas().data(), this->bias.getDeltas().size());
//    Eigen::Matrix<double, Eigen::Dynamic, 1>::ConstAlignedMapType w(this->weightMatrix.getElements().data(),this->weightMatrix.getElements().size());
//    Eigen::Matrix<double, Eigen::Dynamic, 1>::AlignedMapType b(this->bias.getElements().data(),this->bias.getElements().size());
    trainer.optimize(this->weightMatrix);
    trainer.optimize(this->bias);

}

void FullyConnectedLayer::trans(Tensor &tensor, Eigen::MatrixXd &transposed, const Shape &outShape){

    if(tensor.getShape().d == 0){
        transposed = tensor.getElements().transpose();
    }else {
        auto tensor_map = Eigen::TensorMap<Eigen::Tensor<double, 3>> (tensor.getElements().data(),{tensor.getShape().r,tensor.getShape().c,tensor.getShape().d});
        array<int, 2> shuffling({0,2});
        Eigen::Tensor<double, 3> tensor_transposed = tensor_map.shuffle(shuffling);
        transposed = Eigen::Map<Eigen::MatrixXd> (tensor_transposed.data(), outShape.r, outShape.c);
    }
}

FullyConnectedLayer::FullyConnectedLayer(const Tensor &weightMatrix, const Tensor &bias, const Shape &inShape,
                                         const Shape &outShape, bool firstLayer) : weightMatrix(weightMatrix),
                                                                                   bias(bias), inShape(inShape),
                                                                                   outShape(outShape),
                                                                                   first_layer(firstLayer) {}



