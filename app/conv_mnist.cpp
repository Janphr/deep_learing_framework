//
// Created by janphr on 14.05.20.
//

#include <iostream>
#include <Network.h>
#include <FullyConnectedLayer.h>
#include <memory>
#include <SigmoidLayer.h>
#include <SoftmaxLayer.h>
#include <CrossEntropyLayer.h>
#include <SGDTrainer.h>
#include <timer.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <Conv2DLayer.h>

#include "Tensor.h"

void init_tensor_vecs(vector<vector<Tensor>> &vec, int amount_of_datasets, Shape shape, bool pooling){
    for(int a = 0; a < amount_of_datasets; a++){
        MatrixXd m  = MatrixXd::Zero(1, shape.r*shape.c*shape.d),
                d = MatrixXd::Zero(1, shape.r*shape.c*shape.d);
        auto *t = new Tensor(m, d);
        t->getShape() = Shape(shape.r, shape.c, shape.d);
        if(pooling){
            t->getPoolingVec() = *new vector<tuple<int, double>>(shape.c*shape.r*shape.d);
            for(auto &t_ : t->getPoolingVec()){
                t_ = tuple<int, double>(0, -99999.0);
            }
        }
        vector<Tensor> tv = {*t};
        vec.emplace_back(tv);
    }
}
int main(){

    int amount_of_training_datasets = 10000;
    int amount_of_datasets = 9999;
    int amount_filters = 64;
    int amount_epochs = 15;
    int result_for_every_x_dataset = 10;
    float learning_rate = 0.01;

    InputLayer inputLayer;

    int max_pooling_size = 2;
    Shape kernel_shape = {5,5,1};
    Shape in_tensor_shape = {28,28,1};
    Shape t1_tensor_shape = {in_tensor_shape.r - (kernel_shape.r - 1), in_tensor_shape.c - (kernel_shape.c - 1), amount_filters};
    Shape pooling_shape = {(in_tensor_shape.r - (kernel_shape.r - 1))/max_pooling_size, (in_tensor_shape.c - (kernel_shape.c - 1))/max_pooling_size, amount_filters};
    Shape t2_tensor_shape = {1, 10, 1};
    Shape bias_shape1 = {1,10,0};
    Shape weight_shape1 = {t1_tensor_shape.r * t1_tensor_shape.c* t1_tensor_shape.d,10,0};




    MatrixXd kernel_tensor_mat = -1+(ArrayXXd::Random(1, kernel_shape.r* kernel_shape.c * kernel_shape.d * amount_filters)*0.5+0.5)*2;
    MatrixXd kernel_delta = MatrixXd::Zero(1, kernel_shape.r* kernel_shape.c * kernel_shape.d * amount_filters);


    MatrixXd weightsMat1 = -1+(ArrayXXd::Random(weight_shape1.r, weight_shape1.c)*0.5+0.5)*2;
    MatrixXd biasMat1 = MatrixXd::Zero(bias_shape1.r, bias_shape1.c);


    vector<vector<Tensor>> in_tensor, t1, t2, t3, t4, t5;
    vector<Tensor> target_vec;

    inputLayer.convert("../data/train-images.idx3-ubyte", in_tensor, amount_of_training_datasets);
    inputLayer.convert_targets("../data/train-labels.idx1-ubyte", target_vec, amount_of_training_datasets, 10);

    init_tensor_vecs(t1,amount_of_training_datasets, t1_tensor_shape, false);
    init_tensor_vecs(t2,amount_of_training_datasets, t1_tensor_shape, false);
    init_tensor_vecs(t3,amount_of_training_datasets, t2_tensor_shape, false);
    init_tensor_vecs(t4,amount_of_training_datasets, t2_tensor_shape, false);
    init_tensor_vecs(t5,amount_of_training_datasets, Shape(1,1,1), false);

    Tensor kernel_tensor = *new Tensor(kernel_tensor_mat, kernel_delta);
    kernel_tensor.getShape() = kernel_shape;

    shared_ptr<Tensor> weights1 = make_shared<Tensor>(weightsMat1);
    shared_ptr<Tensor> bias1 = make_shared<Tensor>(biasMat1);

    shared_ptr<Layer> conv_layer = make_shared<Conv2DLayer>(kernel_tensor, in_tensor_shape, t1_tensor_shape, kernel_shape.r, amount_filters);
    shared_ptr<Layer> sigmoidActivation = make_shared<SigmoidLayer>();
    shared_ptr<Layer> fullyConnected = make_shared<FullyConnectedLayer>(*weights1, *bias1, Shape(1, t1_tensor_shape.r * t1_tensor_shape.c * t1_tensor_shape.d, 0), Shape(1, 10, 0), false);
    shared_ptr<Layer> softmax = make_shared<SoftmaxLayer>();
    shared_ptr<Layer> crossEntropy = make_shared<CrossEntropyLayer>(target_vec);

    vector<shared_ptr<Layer>> layers = {conv_layer, sigmoidActivation, fullyConnected, softmax, crossEntropy};
    vector<vector<vector<Tensor>>> tensors = {in_tensor, t1, t2, t3, t4, t5};

    shared_ptr<SGDTrainer> trainer = make_shared<SGDTrainer>(amount_epochs, learning_rate);
    unique_ptr<Network> network =  make_unique<Network>(layers, tensors);
    network->set_targets(target_vec);

    unique_ptr<Timer> timer = make_unique<Timer>();

    cout << "Start training with " << amount_of_training_datasets << " data sets.\nLearning rate: " << learning_rate << " for " << amount_epochs << " epochs." << endl;
    timer->start();
    network->train(*trainer);
    double past_time = timer->stop()/1000.0;
    cout << "Training finished in: " << past_time << " seconds." << endl;

    in_tensor.clear(), t1.clear(), t2.clear(), t3.clear(), t4.clear(), t5.clear();
    target_vec.clear();

    inputLayer.convert("../data/t10k-images.idx3-ubyte", in_tensor, amount_of_datasets);
    inputLayer.convert_targets("../data/t10k-labels.idx1-ubyte", target_vec, amount_of_datasets, 10);

    init_tensor_vecs(t1, amount_of_datasets, t1_tensor_shape, false);
    init_tensor_vecs(t2, amount_of_datasets, t1_tensor_shape, false);
    init_tensor_vecs(t3, amount_of_datasets, t2_tensor_shape, false);
    init_tensor_vecs(t4, amount_of_datasets, t2_tensor_shape, false);
    init_tensor_vecs(t5, amount_of_datasets, Shape(1,1,1), false);
    tensors = {in_tensor, t1, t2, t3, t4, t5};

    network->reset_tensors(tensors);
    network->set_data(in_tensor);
    network->set_targets(target_vec);

    cout << "Start classifying with " << amount_of_datasets << " data sets." << endl;    timer->start();
    network->run();
    cout << "Classifying finished in: " << timer->stop()/1000.0 << " seconds." << endl;
    network->print_result(result_for_every_x_dataset);
}

//int main(){
//
//    int amount_of_training_datasets = 100;
//    int amount_of_datasets = 100;
//    int amount_filters = 64;
//    int amount_epochs = 30;
//    int result_for_every_x_dataset = 10;
//    float learning_rate = 0.01;
//
//    InputLayer inputLayer;
//
//    int max_pooling_size = 5;
//    Shape kernel_shape = {4,4,1};
//    Shape in_tensor_shape = {28,28,1};
//    Shape t1_tensor_shape = {in_tensor_shape.r - (kernel_shape.r - 1), in_tensor_shape.c - (kernel_shape.c - 1), amount_filters};
//    Shape pooling_shape = {(in_tensor_shape.r - (kernel_shape.r - 1))/max_pooling_size, (in_tensor_shape.c - (kernel_shape.c - 1))/max_pooling_size, amount_filters};
//    Shape t2_tensor_shape = {1, 10, 1};
//    Shape bias_shape1 = {1,10,0};
//    Shape weight_shape1 = {pooling_shape.r * pooling_shape.c* pooling_shape.d,10,0};
//
//
//
//
//    MatrixXd kernel_tensor_mat = -1+(ArrayXXd::Random(1, kernel_shape.r* kernel_shape.c * kernel_shape.d * amount_filters)*0.5+0.5)*2;
//    MatrixXd kernel_delta = MatrixXd::Zero(1, kernel_shape.r* kernel_shape.c * kernel_shape.d * amount_filters);
//
//
//    MatrixXd weightsMat1 = -1+(ArrayXXd::Random(weight_shape1.r, weight_shape1.c)*0.5+0.5)*2;
//    MatrixXd biasMat1 = MatrixXd::Zero(bias_shape1.r, bias_shape1.c);
//
//
//    vector<vector<Tensor>> in_tensor, t1, t2, t3, t4, t5, t6;
//    vector<Tensor> target_vec;
//
//    inputLayer.convert("../data/train-images.idx3-ubyte", in_tensor, amount_of_training_datasets);
//    inputLayer.convert_targets("../data/train-labels.idx1-ubyte", target_vec, amount_of_training_datasets, 10);
//
//    init_tensor_vecs(t1,amount_of_training_datasets, t1_tensor_shape, false);
//    init_tensor_vecs(t2,amount_of_training_datasets, t1_tensor_shape, false);
//    init_tensor_vecs(t3,amount_of_training_datasets, pooling_shape, true);
//    init_tensor_vecs(t4,amount_of_training_datasets, t2_tensor_shape, false);
//    init_tensor_vecs(t5,amount_of_training_datasets, t2_tensor_shape, false);
//    init_tensor_vecs(t6,amount_of_training_datasets, Shape(1,1,1), false);
//
//    Tensor kernel_tensor = *new Tensor(kernel_tensor_mat, kernel_delta);
//    kernel_tensor.getShape() = kernel_shape;
//
//    shared_ptr<Tensor> weights1 = make_shared<Tensor>(weightsMat1);
//    shared_ptr<Tensor> bias1 = make_shared<Tensor>(biasMat1);
//
//    shared_ptr<Layer> conv_layer = make_shared<Conv2DLayer>(kernel_tensor, in_tensor_shape, t1_tensor_shape, kernel_shape.r, amount_filters);
//    shared_ptr<Layer> sigmoidActivation = make_shared<SigmoidLayer>();
//    shared_ptr<Layer> maxPooling = make_shared<MaxPooling>(max_pooling_size, amount_filters, t1_tensor_shape, pooling_shape);
//    shared_ptr<Layer> fullyConnected = make_shared<FullyConnectedLayer>(*weights1, *bias1, Shape(1, pooling_shape.r * pooling_shape.c * pooling_shape.d, 0), Shape(1, 10, 0), false);
//    shared_ptr<Layer> softmax = make_shared<SoftmaxLayer>();
//    shared_ptr<Layer> crossEntropy = make_shared<CrossEntropyLayer>(target_vec);
//
//    vector<shared_ptr<Layer>> layers = {conv_layer, sigmoidActivation, maxPooling, fullyConnected, softmax, crossEntropy};
//    vector<vector<vector<Tensor>>> tensors = {in_tensor, t1, t2, t3, t4, t5, t6};
//
//    shared_ptr<SGDTrainer> trainer = make_shared<SGDTrainer>(amount_epochs, learning_rate);
//    unique_ptr<Network> network =  make_unique<Network>(layers, tensors);
//    network->set_targets(target_vec);
//
//    unique_ptr<Timer> timer = make_unique<Timer>();
//
//    cout << "Start training with " << amount_of_training_datasets << " data sets.\nLearning rate: " << learning_rate << " for " << amount_epochs << " epochs." << endl;
//    timer->start();
//    network->train(*trainer);
//    double past_time = timer->stop()/1000.0;
//    cout << "Training finished in: " << past_time << " seconds." << endl;
//
//    in_tensor.clear(), t1.clear(), t2.clear(), t3.clear(), t4.clear(), t5.clear(), t6.clear();
//    target_vec.clear();
//
//    inputLayer.convert("../data/t10k-images.idx3-ubyte", in_tensor, amount_of_datasets);
//    inputLayer.convert_targets("../data/t10k-labels.idx1-ubyte", target_vec, amount_of_datasets, 10);
//
//    init_tensor_vecs(t1,amount_of_datasets, t1_tensor_shape, false);
//    init_tensor_vecs(t2,amount_of_datasets, t1_tensor_shape, false);
//    init_tensor_vecs(t3,amount_of_datasets, pooling_shape, true);
//    init_tensor_vecs(t4,amount_of_datasets, t2_tensor_shape, false);
//    init_tensor_vecs(t5,amount_of_datasets, t2_tensor_shape, false);
//    init_tensor_vecs(t6,amount_of_datasets, Shape(1,1,1), false);
//    tensors = {in_tensor, t1, t2, t3, t4, t5, t6};
//
//    network->reset_tensors(tensors);
//    network->set_data(in_tensor);
//    network->set_targets(target_vec);
//
//    cout << "Start classifying with " << amount_of_datasets << " data sets." << endl;    timer->start();
//    network->run();
//    cout << "Classifying finished in: " << timer->stop()/1000.0 << " seconds." << endl;
//    network->print_result(result_for_every_x_dataset);
//}

//
//// Created by janphr on 14.05.20.
////
//
//#include <iostream>
//#include <Network.h>
//#include <FullyConnectedLayer.h>
//#include <memory>
//#include <SigmoidLayer.h>
//#include <SoftmaxLayer.h>
//#include <CrossEntropyLayer.h>
//#include <SGDTrainer.h>
//#include <timer.h>
//#include <opencv2/core.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <Conv2DLayer.h>
//
//#include "Tensor.h"
//
//void init_tensor_vecs(vector<vector<Tensor>> &vec, int amount_of_datasets, Shape shape){
//    for(int a = 0; a < amount_of_datasets; a++){
//        MatrixXd m  = MatrixXd::Zero(1, shape.r*shape.c*shape.d),
//                d = MatrixXd::Zero(1, shape.r*shape.c*shape.d);
//        auto *t = new Tensor(m, d);
//        t->getShape() = Shape(shape.r, shape.c, shape.d);
//        vector<Tensor> tv = {*t};
//        vec.emplace_back(tv);
//    }
//}
//int main(){
//
//    int amount_of_training_datasets = 5;
//    int amount_of_datasets = 9999;
//    int amount_filters = 64;
//    int amount_epochs = 1;
//    int result_for_every_x_dataset = 100;
//    float learning_rate = 0.01;
//
//    InputLayer inputLayer;
//
//    Shape kernel_shape = {3,3,1};
//    Shape in_tensor_shape = {28,28,1};
//    Shape t1_tensor_shape = {in_tensor_shape.r - (kernel_shape.r - 1), in_tensor_shape.c - (kernel_shape.c - 1), amount_filters};
//    Shape t2_tensor_shape = {1, 26*26, 1};
//    Shape t3_tensor_shape = {1, 10, 1};
//    Shape bias_shape1 = {1,t2_tensor_shape.c,0};
//    Shape weight_shape1 = {t1_tensor_shape.r * t1_tensor_shape.c* t1_tensor_shape.d,t2_tensor_shape.c,0};
//    Shape bias_shape2 = {1,10,0};
//    Shape weight_shape2 = {t2_tensor_shape.c,10,0};
//
//
//
//
//    MatrixXd kernel_tensor_mat = -1+(ArrayXXd::Random(1, kernel_shape.r* kernel_shape.c * kernel_shape.d * amount_filters)*0.5+0.5)*2;
//    MatrixXd kernel_delta = MatrixXd::Zero(1, kernel_shape.r* kernel_shape.c * kernel_shape.d * amount_filters);
//
//
//    MatrixXd weightsMat1 = -1+(ArrayXXd::Random(weight_shape1.r, weight_shape1.c)*0.5+0.5)*2;
//    MatrixXd biasMat1 = MatrixXd::Zero(bias_shape1.r, bias_shape1.c);
//
//    MatrixXd weightsMat2 = -1+(ArrayXXd::Random(weight_shape2.r, weight_shape2.c)*0.5+0.5)*2;
//    MatrixXd biasMat2 = MatrixXd::Zero(bias_shape2.r, bias_shape2.c);
//
//
//    vector<vector<Tensor>> in_tensor, t1, t2, t3, t4, t5, t6, t7;
//    vector<Tensor> target_vec;
//
//    inputLayer.convert("../data/train-images.idx3-ubyte", in_tensor, amount_of_training_datasets);
//    inputLayer.convert_targets("../data/train-labels.idx1-ubyte", target_vec, amount_of_training_datasets, 10);
//
//    init_tensor_vecs(t1,amount_of_training_datasets, t1_tensor_shape);
//    init_tensor_vecs(t2,amount_of_training_datasets, t1_tensor_shape);
//    init_tensor_vecs(t3,amount_of_training_datasets, t2_tensor_shape);
//    init_tensor_vecs(t4,amount_of_training_datasets, t2_tensor_shape);
//    init_tensor_vecs(t5,amount_of_training_datasets, t3_tensor_shape);
//    init_tensor_vecs(t6,amount_of_training_datasets, t3_tensor_shape);
//    init_tensor_vecs(t7,amount_of_training_datasets, Shape(1,1,1));
//
//    Tensor kernel_tensor = *new Tensor(kernel_tensor_mat, kernel_delta);
//    kernel_tensor.getShape() = kernel_shape;
//
//    shared_ptr<Tensor> weights1 = make_shared<Tensor>(weightsMat1);
//    shared_ptr<Tensor> bias1 = make_shared<Tensor>(biasMat1);
//
//    shared_ptr<Tensor> weights2 = make_shared<Tensor>(weightsMat2);
//    shared_ptr<Tensor> bias2 = make_shared<Tensor>(biasMat2);
//
//    shared_ptr<Layer> conv_layer = make_shared<Conv2DLayer>(kernel_tensor, in_tensor_shape, t1_tensor_shape, kernel_shape.r, amount_filters);
//    shared_ptr<Layer> sigmoidActivation = make_shared<SigmoidLayer>();
//    shared_ptr<Layer> fullyConnected = make_shared<FullyConnectedLayer>(*weights1, *bias1, Shape(1, t1_tensor_shape.r * t1_tensor_shape.c * t1_tensor_shape.d, 0), Shape(1, t2_tensor_shape.c, 0), false);
//    shared_ptr<Layer> sigmoidActivation2 = make_shared<SigmoidLayer>();
//    shared_ptr<Layer> fullyConnected2 = make_shared<FullyConnectedLayer>(*weights2, *bias2, Shape(1, t2_tensor_shape.c, 0), Shape(1, 10, 0), false);
//    shared_ptr<Layer> softmax = make_shared<SoftmaxLayer>();
//    shared_ptr<Layer> crossEntropy = make_shared<CrossEntropyLayer>(target_vec);
//
//    vector<shared_ptr<Layer>> layers = {conv_layer, sigmoidActivation, fullyConnected, sigmoidActivation2, fullyConnected2, softmax, crossEntropy};
//    vector<vector<vector<Tensor>>> tensors = {in_tensor, t1, t2, t3, t4, t5, t6, t7};
//
//    shared_ptr<SGDTrainer> trainer = make_shared<SGDTrainer>(amount_epochs, learning_rate);
//    unique_ptr<Network> network =  make_unique<Network>(layers, tensors);
//    network->set_targets(target_vec);
//
//    unique_ptr<Timer> timer = make_unique<Timer>();
//
//    cout << "Start training with " << amount_of_training_datasets << " data sets.\nLearning rate: " << learning_rate << " for " << amount_epochs << " epochs." << endl;
//    timer->start();
//    network->train(*trainer);
//    double past_time = timer->stop()/1000.0;
//    cout << "Training finished in: " << past_time << " seconds." << endl;
//}