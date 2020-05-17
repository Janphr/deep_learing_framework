//
// Created by janphr on 13.05.20.
//

#include <Network.h>
#include <FullyConnectedLayer.h>
#include <SigmoidLayer.h>
#include <SoftmaxLayer.h>
#include <CrossEntropyLayer.h>
#include <iomanip>
#include <timer.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv/cv.hpp>
#include <numeric>

using namespace cv;

int main(){

    InputLayer inputLayer;

    Shape weight_shape1 = {28*28,32,0};  //28*28 x 32
    Shape weight_shape2 = {32,10,0};  //32 x 10
    Shape bias_shape1 = {1,32,0};    //1 x 32
    Shape bias_shape2 = {1,10,0};    //1 x 10

    vector<vector<Tensor>> t1_vec{{*new Tensor()}}, t2_vec{{*new Tensor()}}, t3_vec{{*new Tensor()}}, t4_vec{{*new Tensor()}}, t5_vec{{*new Tensor()}}, t6_vec{{*new Tensor()}};
    vector<Tensor> target_vec{*new Tensor(*new MatrixXd(1,10))};

    MatrixXd weightsMat1 = -1+(ArrayXXd::Random(weight_shape1.r, weight_shape1.c)*0.5+0.5)*2;
    MatrixXd weightsMat2 = -1+(ArrayXXd::Random(weight_shape2.r, weight_shape2.c)*0.5+0.5)*2;
    MatrixXd biasMat1 = MatrixXd::Zero(bias_shape1.r, bias_shape1.c);
    MatrixXd biasMat2 = MatrixXd::Zero(bias_shape2.r, bias_shape2.c);

    cv::FileStorage fs("../data/training_data.xml", cv::FileStorage::READ);
    if (!fs.isOpened())
        throw runtime_error("Could not open the training data file: ");
    cv::Mat dst;
    fs["weight_matrix1"] >> dst;
    cv2eigen(dst, weightsMat1);
    fs["weight_matrix2"] >> dst;
    cv2eigen(dst, weightsMat2);
    fs["bias_matrix1"] >> dst;
    cv2eigen(dst, biasMat1);
    fs["bias_matrix2"] >> dst;
    cv2eigen(dst, biasMat2);

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

    unique_ptr<Network> network =  make_unique<Network>(layers, tensors);
    network->set_targets(target_vec);

    VideoCapture cap;
    if(!cap.open(0))
        return 0;

    int array_size = 10;
    int detected_number[array_size] = {0};
    double probability[array_size] = {0};
    int counter = 9;
    double sum_p = 0;
    int sum_d = 0;
    MatrixXd eigen_mat;
    Mat frame, grey, edited_grey;
    string msg;
    while(true){


        cap >> frame;
        if( frame.empty() ) break;

        cvtColor(frame, grey, CV_BGR2GRAY);
        Rect target_area(Point(frame.cols/2 - 112, frame.rows/2 - 112), Point(frame.cols/2 + 112, frame.rows/2 + 112));

        rectangle(frame, target_area , Scalar(0,255,0), 2);
        grey = grey(target_area);
        resize(grey, grey, Size(28,28));

        adaptiveThreshold(grey, grey, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,51,5);

        bitwise_not(grey, grey);

        rotate(grey, edited_grey, ROTATE_90_COUNTERCLOCKWISE);
        flip(edited_grey, edited_grey, 0);


        cv2eigen(edited_grey, eigen_mat);

        eigen_mat.resize(1, 28*28);
        for(int i = 0; i < eigen_mat.cols(); i++){
            eigen_mat.coeffRef(i) /= 255;
        }


        t1_vec[0][0].setElements(eigen_mat);

        network->set_data(t1_vec);
        network->detect(detected_number[counter], probability[counter]);

        resize(grey, grey, Size(224,224));
        cvtColor(grey, grey, CV_GRAY2BGR);
        grey.copyTo(frame(Rect(frame.cols/2 - 112,frame.rows/2 - 112,grey.cols, grey.rows)));

        counter++;
        if(counter == 10){
            msg = format("Number %d detected, with %f probability.",(int)std::accumulate(detected_number,detected_number+10, sum_d)/counter,
                                100*std::accumulate(probability,probability+10, sum_p)/counter);

            counter = 0;
            sum_p = 0;
            sum_d = 0;
        }

        Size textSize = getTextSize(msg, 1, 1, 1, 0);
        Point textOrigin = Size(frame.cols/2 - textSize.width/2, 10);
        putText(frame, msg, textOrigin, 1, 1, Scalar(255,255,255));
        imshow("Number detector.", frame);

        if( waitKey(10) == 27 ) break;
    }

}