//
// Created by janphr on 29.04.20.
//

#include <iomanip>
#include "Network.h"
#include <omp.h>

void Network::run() {
    int layer_size = this->layers.size();
    MatrixXf::Index maxIndex[2];
    double accuracy = 0;
    for (int i = 0; i < this->tensors[0].size(); i++) { //over all data tensors
        for (int j = 0; j < layer_size; j++) {       //over all layers
            this->layers[j]->forward(this->tensors[j][i], this->tensors[j + 1][i], 0);
        }

        for (int j = 0; j < this->tensors[layer_size - 1][i].size(); j++) {
            this->tensors[layer_size - 1][i][j].getElements().row(0).maxCoeff(&maxIndex[0]);
            this->targets[i].getElements().row(0).maxCoeff(&maxIndex[1]);
            accuracy += maxIndex[0] == maxIndex[1] ? 1 : 0;
        }
    }
    cout << "Final Accuracy: " << accuracy / this->tensors[0].size() << endl;
}

void Network::train(SGDTrainer &trainer) {
    int counter = 0;
    double avg_loss = 0;
    double accuracy = 0;
    int layer_size = this->layers.size();
    MatrixXf::Index maxIndex[2];
    omp_set_num_threads(8);
    while (trainer.getAmountEpochs() > counter) {

//#pragma omp parallel for reduction(+:avg_loss,accuracy)
        for (int i = 0; i < this->tensors[0].size(); i++) { //over all data tensors
            for (int j = 0; j < layer_size; j++) {       //over all layers
                this->layers[j]->forward(this->tensors[j][i], this->tensors[j + 1][i], i);
            }

            reset_deltas(i);

            for (auto t : this->tensors[layer_size][i]) {
                avg_loss += t.getElements().coeffRef(0);
            }

            for (int j = 0; j < this->tensors[layer_size - 1][i].size(); j++) {
                this->tensors[layer_size - 1][i][j].getElements().row(0).maxCoeff(&maxIndex[0]);
                this->targets[i].getElements().row(0).maxCoeff(&maxIndex[1]);
                accuracy += maxIndex[0] == maxIndex[1] ? 1 : 0;
            }

            for (int j = layer_size - 1; j >= 0; j--) {
                this->layers[j]->backward(this->tensors[j + 1][i], this->tensors[j][i], i);
            }

            for (int j = 0; j < layer_size; j++) {
//#pragma omp critical
                this->layers[j]->update(trainer);
            }
        }


        cout << "Epoche: " << counter + 1 << " Avg loss: " << avg_loss / this->tensors[0].size() << " Accuracy: "
             << accuracy / this->tensors[0].size() << endl;


        counter++;
        avg_loss = 0;
        accuracy = 0;
    }
}

Network::Network(vector<shared_ptr<Layer>> &layers, vector<vector<vector<Tensor>>> &tensors) {
    this->layers = layers;
    this->tensors = tensors;
}

void Network::set_data(vector<vector<Tensor>> &data) {
    this->tensors[0] = data;
}

void Network::set_targets(vector<Tensor> &targets) {
    this->targets = targets;
}

void Network::print_result(int every_x_dataset) {
    int end_layer_size = this->targets[0].getElements().size();
    for (int i = 0; i < this->targets.size() - every_x_dataset; i += every_x_dataset) {
        cout << setw(26) << "Softmax probability: ";
        for (int j = 0; j < end_layer_size; j++) {
            cout << setw(15) << this->tensors[this->tensors.size() - 2][i][0].getElements().coeffRef(j);
        }
        cout << endl << setw(26) << "Target values:";
        for (int j = 0; j < end_layer_size; j++) {
            cout << setw(15) << this->targets[i].getElements().coeffRef(j);
        }
        cout << endl << setw(26) << "Loss:";
        cout << this->tensors[this->tensors.size() - 1][i][0].getElements();

        cout << endl << "--------------------------------------------------"
                        "--------------------------------------------------"
                        "--------------------------------------------------"
                        "--------------------------------------------------" << endl;
    }
}


void Network::reset_tensors(vector<vector<vector<Tensor>>> &tensors) {
    this->tensors = tensors;
    cout << tensors[0].size() << endl;
    cout << this->tensors[0].size() << endl;
}

void Network::reset_deltas(int i) {
    for (const auto &v : this->tensors) {
        for (auto t : v[i]) {
            t.getDeltas().setZero();
        }
    }
}

//void Network::print_debug(int every_x_dataset) {
//    while (trainer->getAmountEpochs() > counter) {
//        fullyConnected1->forward(t1_vec, t2_vec);
////        cout << "fullyConnected1: " << t2_vec[0].getElements() << endl;
//        sigmoidActivation->forward(t2_vec, t3_vec);
////        cout << "sigmoidActivation:" << t3_vec[0].getElements() << endl;
//        fullyConnected2->forward(t3_vec, t4_vec);
////        cout << "fullyConnected2:" << t4_vec[0].getElements() << endl;
//        softmax->forward(t4_vec, t5_vec);
//        cout << "softmax:" << t5_vec[0].getElements() << endl;
//        crossEntropy->forward(t5_vec, t6_vec);
//
////        cout << "target:" << t6_vec[0].getElements() << endl;
//
//        crossEntropy->backward(t6_vec, t5_vec);
////        cout << "crossEntropy:" << t5_vec[0].getDeltas() << endl;
//        softmax->backward(t5_vec, t4_vec);
////        cout << "softmax:" << t4_vec[0].getDeltas() << endl;
//        fullyConnected2->backward(t4_vec, t3_vec);
////        cout << "fullyConnected2:" << t3_vec[0].getDeltas() << endl;
//        sigmoidActivation->backward(t3_vec, t2_vec);
////        cout << "sigmoidActivation:" << t2_vec[0].getDeltas() << endl;
//        fullyConnected1->backward(t2_vec, t1_vec);
////        cout << "fullyConnected1:" << t1_vec[0].getDeltas() << endl;
//
//        trainer->optimize(t1_vec);
//        counter++;
////        cout << "-------------------------------" << endl;
//    }
//}
