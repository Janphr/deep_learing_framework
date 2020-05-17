//
// Created by janphr on 10.05.20.
//

#include <Network.h>
#include <FullyConnectedLayer.h>
#include <memory>
#include <opencv2/core.hpp>
#include <Conv2DLayer.h>

#include "Tensor.h"


int main() {


    InputLayer inputLayer;

    int max_pooling_size = 1;

    Shape kernel_shape = {2,2,2};
    Shape in_tensor_shape = {4,3,2};
    Shape out_tensor_shape = {3,2,2};


    vector<vector<Tensor>> in_tensor, out_tensor, kernel_tensor;

    vector<vector<double>> in_tensor_data {{0.1, -0.2, 0.5, 0.6, 1.2, 1.4, 1.6, 2.2, 0.01, 0.2, -0.3, 4.0, 0.9, 0.3, 0.5, 0.65, 1.1, 0.7, 2.2, 4.4, 3.2, 1.7, 6.3, 8.2}},
    out_tensor_data {{0,0,0,0,0,0,0,0,0,0,0,0}},
    kernel_tensor_data {{0.1, -0.2, 0.3, 0.4, 0.7, 0.6, 0.9, -1.1, 0.37, -0.9, 0.32, 0.17, 0.9, 0.3, 0.2, -0.7}};

    MatrixXd kernel_tensor_mat = MatrixXd::Zero(1, kernel_shape.r* kernel_shape.c * kernel_shape.d * 2);

    MatrixXd out_tensor_delta = MatrixXd::Zero(1, out_tensor_shape.r* out_tensor_shape.c * out_tensor_shape.d);
    out_tensor_delta << 0.1, 0.33, -0.6, -0.25, 1.3, 0.01, -0.5, 0.2, 0.1, -0.8, 0.81, 1.1;
    MatrixXd in_tensor_delta = MatrixXd::Zero(1, in_tensor_shape.r* in_tensor_shape.c * in_tensor_shape.d);
    Eigen::MatrixXd kernel_delta = Eigen::MatrixXd::Zero(1, kernel_shape.r*kernel_shape.r*kernel_shape.r*kernel_shape.r);

    inputLayer.convert(in_tensor_data, in_tensor);
    inputLayer.convert(out_tensor_data, out_tensor);
    inputLayer.convert(kernel_tensor_data, kernel_tensor);

    in_tensor[0][0].getShape() = in_tensor_shape;
    in_tensor[0][0].setDeltas(in_tensor_delta);
    out_tensor[0][0].getShape() = out_tensor_shape;
    out_tensor[0][0].setDeltas(out_tensor_delta);
    kernel_tensor[0][0].getShape() = kernel_shape;
    kernel_tensor[0][0].setDeltas(kernel_tensor_mat);
    out_tensor[0][0].getPoolingVec() = *new vector<tuple<int, double>>(out_tensor_shape.c*out_tensor_shape.r*out_tensor_shape.d);
    for(auto &t : out_tensor[0][0].getPoolingVec()){
        t = tuple<int, double>(0, -99999.0);
    }

    cout << get<1>(out_tensor[0][0].getPoolingVec().at(0)) << endl;

    shared_ptr<Layer> conv_layer = make_shared<Conv2DLayer>(kernel_tensor[0][0], in_tensor_shape, out_tensor_shape, kernel_shape.r, 2);

    conv_layer->forward(in_tensor[0], out_tensor[0], 0);

    conv_layer->backward(out_tensor[0], in_tensor[0],0);

}

