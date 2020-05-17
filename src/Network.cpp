//
// Created by janphr on 29.04.20.
//

#include <iomanip>
#include "Network.h"
#include <omp.h>
#include <timer.h>

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

void Network::detect(int &detected_number, double &probability) {
    int layer_size = this->layers.size();
    MatrixXf::Index maxIndex;

    for (int i = 0; i < this->tensors[0].size(); i++) { //over all data tensors
        for (int j = 0; j < layer_size; j++) {       //over all layers
            this->layers[j]->forward(this->tensors[j][i], this->tensors[j + 1][i], 0);
        }

        for (auto & j : this->tensors[layer_size - 1][i]) {
            j.getElements().row(0).maxCoeff(&maxIndex);
            detected_number = maxIndex;
            probability = j.getElements().row(0).coeffRef(detected_number);
        }
    }
    cout << "Detected: " << detected_number << " with " << probability*100 << "% certainty" << endl;
}

void Network::train(SGDTrainer &trainer) {
    int counter = 0;
    double avg_loss = 0;
    double accuracy = 0;
    int layer_size = this->layers.size();
    int tensor_size = this->tensors[0].size();;
    MatrixXf::Index maxIndex[2];
    omp_set_num_threads(8);
    unique_ptr<Timer> timer = make_unique<Timer>();
    while (trainer.getAmountEpochs() > counter) {
//#pragma omp parallel for reduction(+:avg_loss,accuracy)
        cout << "Progress of " << counter + 1 << ". epochs:" << endl;
        for (int i = 0; i < tensor_size; i++) { //over all data tensors
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
            cout << '\r' << ((1.0*i)/tensor_size) * 100 << "%";
        }


        cout << endl << "Epochs: " << counter + 1 << " Avg loss: " << avg_loss / tensor_size << " Accuracy: "
             << accuracy / tensor_size << endl;


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