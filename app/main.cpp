#include <iostream>
#include <Network.h>
#include <FullyConnectedLayer.h>
#include <memory>
#include <SigmoidLayer.h>
#include <SoftmaxLayer.h>
#include <CrossEntropyLayer.h>
#include <SGDTrainer.h>
#include <iomanip>
#include <timer.h>

#include "Tensor.h"

void init_tensor_vecs(vector<vector<Tensor>> &vec, int amount_of_datasets){
    for(int a = 0; a < amount_of_datasets; a++){
        auto *t = new Tensor();
        vector<Tensor> tv = {*t};
        vec.emplace_back(tv);
    }
}

int main() {
//    std::cout << "Hello, World!" << std::endl;

    int amount_of_training_datasets = 1000;
    int amount_of_datasets = 9999;
    int amount_epochs = 1;
    int result_for_every_x_dataset = 100;
    float learning_rate = 0.01;
    Shape weight_shape1 = {28*28,32,0};  //28*28 x 32
    Shape weight_shape2 = {32,10,0};  //32 x 10
    Shape bias_shape1 = {1,32,0};    //1 x 32
    Shape bias_shape2 = {1,10,0};    //1 x 10


//    vector<vector<double>> start_data = {{0.4183, 0.5209, 0.0291}};
//    vector<vector<double>> target_data = {{0.7095, 0.0942}};

    shared_ptr<Tensor> t1 = make_shared<Tensor>(),
            t_target = make_shared<Tensor>();

    vector<vector<Tensor>> t1_vec, t2_vec, t3_vec, t4_vec, t5_vec, t6_vec;
    vector<Tensor> target_vec;

    InputLayer inputLayer;
//    inputLayer.convert(start_data, t1_vec);
//    inputLayer.convert_targets(target_data, target_vec);

    inputLayer.convert("../data/train-images.idx3-ubyte", t1_vec, amount_of_training_datasets);
    inputLayer.convert_targets("../data/train-labels.idx1-ubyte", target_vec, amount_of_training_datasets, 10);

    init_tensor_vecs(t2_vec, amount_of_training_datasets);
    init_tensor_vecs(t3_vec, amount_of_training_datasets);
    init_tensor_vecs(t4_vec, amount_of_training_datasets);
    init_tensor_vecs(t5_vec, amount_of_training_datasets);
    init_tensor_vecs(t6_vec, amount_of_training_datasets);

//    cout << t1_vec[1].getElements() << endl;

//    for(auto n : target_vec)
//        cout << n.getElements() << endl;

    MatrixXd weightsMat1 = -1+(ArrayXXd::Random(weight_shape1.r, weight_shape1.c)*0.5+0.5)*2;
    MatrixXd weightsMat2 = -1+(ArrayXXd::Random(weight_shape2.r, weight_shape2.c)*0.5+0.5)*2;
//    weightsMat1 = -1+(ArrayXXd::Random(weight_shape1.r, weight_shape1.c)*0.5+0.5)*2;
//    weightsMat2 = -1+(ArrayXXd::Random(weight_shape2.r, weight_shape2.c)*0.5+0.5)*2;



//    weightsMat1 << -0.5057, 0.3987, -0.8943,
//            0.3356, 0.1673, 0.8321,
//            -0.3485, -0.4597, -0.1121;
//
//    weightsMat2 << 0.4047, 0.9563,
//            -0.8192, -0.1274,
//            0.3662, -0.7252;

//    cout << "Weights1: " << weightsMat1 << endl;
//    cout << "Weights2: " << weightsMat2 << endl;



    MatrixXd biasMat1 = MatrixXd::Zero(bias_shape1.r, bias_shape1.c);
    MatrixXd biasMat2 = MatrixXd::Zero(bias_shape2.r, bias_shape2.c);

//    cv::FileStorage fs("../data/training_data.xml", cv::FileStorage::READ);
//    if (!fs.isOpened())
//        throw runtime_error("Could not open the training data file: ");
//    cv::Mat dst;
//    fs["weight_matrix1"] >> dst;
//    cv2eigen(dst, weightsMat1);
//    fs["weight_matrix2"] >> dst;
//    cv2eigen(dst, weightsMat2);
//    fs["bias_matrix1"] >> dst;
//    cv2eigen(dst, biasMat1);
//    fs["bias_matrix2"] >> dst;
//    cv2eigen(dst, biasMat2);

    shared_ptr<Tensor> weights1 = make_shared<Tensor>(weightsMat1);
    shared_ptr<Tensor> weights2 = make_shared<Tensor>(weightsMat2);

    shared_ptr<Tensor> bias1 = make_shared<Tensor>(biasMat1);
    shared_ptr<Tensor> bias2 = make_shared<Tensor>(biasMat2);

    shared_ptr<Layer> fullyConnected1 = make_shared<FullyConnectedLayer>(*weights1, *bias1, Shape(1, weight_shape1.r, 0), Shape(1, weight_shape1.c, 0), true);
    shared_ptr<Layer> sigmoidActivation = make_shared<SigmoidLayer>();
    shared_ptr<Layer> fullyConnected2 = make_shared<FullyConnectedLayer>(*weights2, *bias2, Shape(1, weight_shape2.r, 0), Shape(1, weight_shape2.c, 0), false);
    shared_ptr<Layer> softmax = make_shared<SoftmaxLayer>();
    shared_ptr<Layer> crossEntropy = make_shared<CrossEntropyLayer>(target_vec);

    vector<shared_ptr<Layer>> layers = {fullyConnected1, sigmoidActivation, fullyConnected2, softmax, crossEntropy};
    vector<vector<vector<Tensor>>> tensors = {t1_vec, t2_vec, t3_vec, t4_vec, t5_vec, t6_vec};

    shared_ptr<SGDTrainer> trainer = make_shared<SGDTrainer>(amount_epochs, learning_rate);
    unique_ptr<Network> network =  make_unique<Network>(layers, tensors);
    network->set_targets(target_vec);

    unique_ptr<Timer> timer = make_unique<Timer>();

    cout << "Start training with " << amount_of_training_datasets << " data sets.\nLearning rate: " << learning_rate << " for " << amount_epochs << " epochs." << endl;
    timer->start();
    network->train(*trainer);
    double past_time = timer->stop()/1000.0;
    cout << "Training finished in: " << past_time << " seconds." << endl;
    network->print_result(result_for_every_x_dataset);
//
//    cv::FileStorage fs("../data/training_data.xml", cv::FileStorage::WRITE);
//    if (!fs.isOpened())
//        throw runtime_error("Could not open the training data file: ");
//    cv::Mat dst;
//
//    fs << "training_time" << past_time;
//    fs << "training_set_count" << amount_of_training_datasets;
//    eigen2cv(weights1->getElements(), dst);
//    fs << "weight_matrix1" << dst;
//    eigen2cv(weights2->getElements(), dst);
//    fs << "weight_matrix2" << dst;
//    eigen2cv(bias1->getElements(), dst);
//    fs << "bias_matrix1" << dst;
//    eigen2cv(bias2->getElements(), dst);
//    fs << "bias_matrix2" << dst;
//
//    fs.release();



    t1_vec.clear(), t2_vec.clear(), t3_vec.clear(), t4_vec.clear(), t5_vec.clear(), t6_vec.clear();
    target_vec.clear();

    inputLayer.convert("../data/t10k-images.idx3-ubyte", t1_vec, amount_of_datasets);
    inputLayer.convert_targets("../data/t10k-labels.idx1-ubyte", target_vec, amount_of_datasets, 10);

    init_tensor_vecs(t2_vec, amount_of_datasets);
    init_tensor_vecs(t3_vec, amount_of_datasets);
    init_tensor_vecs(t4_vec, amount_of_datasets);
    init_tensor_vecs(t5_vec, amount_of_datasets);
    init_tensor_vecs(t6_vec, amount_of_datasets);
    tensors = {t1_vec, t2_vec, t3_vec, t4_vec, t5_vec, t6_vec};
    cout << t1_vec[0][0].getElements() << endl;

    network->reset_tensors(tensors);
    network->set_data(t1_vec);
    network->set_targets(target_vec);

    cout << "Start classifying with " << amount_of_datasets << " data sets." << endl;    timer->start();
    network->run();
    cout << "Classifying finished in: " << timer->stop()/1000.0 << " seconds." << endl;
    network->print_result(result_for_every_x_dataset);

    return 0;
}
