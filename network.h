#ifndef NETWORK_H
#define NETWORK_H
#include "smalldatafile.h"

typedef struct Neuron_t{
    float act;
    float z;
    float bias;
    float * out_weights;
    
    float dact;
    float dbias;
    float dz;
    float * dw;
} Neuron;

typedef struct Layer_t{
    Neuron * neurons;
    int num_neu;
} Layer;

typedef struct Network_t{
    Layer * layers;
    int num_layer;

} Network;

Layer create_layer(int num_neuron);
Neuron create_neuron(int num_output_neuron,int j);
void initialize_nabla(Network * network);
Network* create_Network();
void feedforward(Network* network,dataset_t * datasets,int p);
int mini_batch(dataset_t * datasets, dataset_t * batch, int size, int number);
void backprop (dataset_t * mini_batch, Network * network,int p);
void update_mini_batch(dataset_t * mini_batch,Network * network);
int compute_acc(Network * network,dataset_t * test_data);
void SGD(dataset_t * datasets,Network * network,dataset_t * test_data);
#endif