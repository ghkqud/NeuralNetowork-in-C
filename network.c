/**********************
 * Program Name: network.c (Module to implement feedforward and back propagation)
 * Brief Description : Create a Network structure that consists of Neurons.
 *      Implement feedforward propagation and calculate Gradient by using backpropagation
 *      algorithm.
 *  
 *Usage Instructions: This module is automatically executed in main.c

 Author: C.H.B
 Created on : July 30,2021
 Last modified on : August 5, 2021
 */

// Include files - include the basic set of definitions and functions
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#define eps 1.0
#define epoch 1000
#define batch_size 10
#define num_layers 3
#define DATASIZE 110

// Include files - include network module and get 5X5 data module
#include "network.h"
#include "smalldatafile.h"

// number of neurons. Define num_neuron by global variable.
int num_neuron[3] = {25,30,10};

//derivative of sigmoid function
float sigmoid_prime(float z){
    float result;
    result =(1/(1+exp(-z))) * (1-(1/(1+exp(-z))));
    return result;
}
// allocate layer structure
Layer create_layer(int num_neuron){

    Layer lay;
    lay.num_neu = num_neuron;
    lay.neurons = (Neuron *)malloc(num_neuron * sizeof(Neuron));
    //printf("%d\n",lay.neurons);
    return lay;
}
// create neuron's value. Allocate & initialize weights and bias
Neuron create_neuron(int num_output_neuron,int j){
    Neuron neu;
    neu.act = 0;
    neu.z = 0;
    neu.bias = 0;
    neu.out_weights = (float *)malloc(num_output_neuron * sizeof(float));

    neu.dact = 0;
    neu.dz = 0;
    neu.dbias = 0;
    neu.dw = (float *)malloc(num_output_neuron * sizeof(float));
    if (j>0){
            neu.bias = (2*rand() / RAND_MAX)-1;
            //neu.bias = normal(1,0);
            //printf("bias  %f  \n",neu.bias);
        }
    // Input layer doesn't have bias, so adds j.
    for (int i=0; i<num_output_neuron;i++){
        if (j<2){
        neu.out_weights[i] = (2*(float)rand() / (float)RAND_MAX)-1;
        //neu.out_weights[i] = normal(1,0);
        neu.dw[i] = 0;
        //printf("weight : %f  \n",neu.out_weights[i]);
        }
        
    }

    return neu;
}
//initialize nabla_w or nabla_b before gradient update.
void initialize_nabla(Network * network){
    int i,j,k;
    for (int i=0; i<num_layers;i++){
        for(int j=0;j<num_neuron[i];j++){
            network->layers[i].neurons[j].dbias=0;
            network->layers[i].neurons[j].dz = 0;
            network->layers[i].neurons[j].dact=0;
            if (i<num_layers-1){
                memset(network->layers[i].neurons[j].dw,0,sizeof(float)*num_neuron[i+1]);
            }
        }
    }
}
//create network that consists of num_layers(=3) layers.
Network* create_Network()
{
    Network* network = (Network *)malloc(sizeof(Network));
    int i = 0;
    int j = 0;
    network->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    if (network->layers == NULL){
        printf("failed to memory allocation\n");
    }
    // weight and bias is created and initialization   
    for (i=0;i<num_layers;i++){
        network->layers[i] = create_layer(num_neuron[i]);
        for (j=0;j<num_neuron[i];j++){
            if ( (i+1) <= num_layers)
            {
                network->layers[i].neurons[j] = create_neuron(num_neuron[i+1],i);
            }  
            
        }

    }
    // weight and bias initialization

    return network;

}
//feedforward input layer's data to output layer's output 
void feedforward(Network* network,dataset_t * datasets,int p){
    int i,j,k;
    for (int i=0;i<num_neuron[0];i++){
        network->layers[0].neurons[i].act = datasets->images[p].pixels[i];

    }

    for(int i=1;i<num_layers;i++)
    {   
        for(int j=0;j<num_neuron[i];j++)
        {
            network->layers[i].neurons[j].z = network->layers[i].neurons[j].bias;

            for(int k=0;k<num_neuron[i-1];k++)
            {
                // z^l = w^l-1 * a^l-1 + b^l
                network->layers[i].neurons[j].z  += ((network->layers[i-1].neurons[k].out_weights[j])* (network->layers[i-1].neurons[k].act));
            }        
            // Sigmoid Activation function
            network->layers[i].neurons[j].act = 1/(1+exp(-(network->layers[i].neurons[j].z)));
                       
            
        }
       
    }
 
}

// Return mini batch's address. batch_address move by using start_offset  
int mini_batch(dataset_t * datasets, dataset_t * batch, int size, int number){
    int start_offset;

    start_offset = size * number;

    if(start_offset > datasets->size){
        return 0;
    }
    batch->images = &(datasets->images[start_offset]);
    batch->labels = &(datasets->labels[start_offset]);
    batch->size = size;
    

    return 1;
}
// Return nabla_w, nable_b representing the gradient for the cost function C. 
void backprop (dataset_t * mini_batch, Network * network,int p){
    feedforward(network,mini_batch,p);
    initialize_nabla(network);
    //output layer backprop
    for (int i=0;i<num_neuron[num_layers -1];i++){
        network->layers[num_layers -1].neurons[i].dz = (network->layers[num_layers -1].neurons[i].act - mini_batch->labels[p].desired_out[i])*sigmoid_prime(network->layers[num_layers-1].neurons[i].z);
        //printf("%f\n", (network->layers[2].neurons[i].z));
        network->layers[num_layers -1].neurons[i].dbias = network->layers[num_layers -1].neurons[i].dz;
        for (int j=0;j<num_neuron[num_layers - 2];j++){
            
            network->layers[num_layers -2].neurons[j].dw[i] = (network->layers[num_layers -1].neurons[i].dz) * (network->layers[num_layers -2].neurons[j].act);
            network->layers[num_layers -2].neurons[j].dz = network->layers[num_layers -2].neurons[j].dz \
                    + ((network->layers[num_layers-1].neurons[i].dz) * (network->layers[num_layers-2].neurons[j].out_weights[i])\
                    * sigmoid_prime(network->layers[num_layers-2].neurons[j].z));

        }
    }
    //hidden layer backprop
    for (int i=num_layers-2;i>0;i--){
        for (int j=0;j<num_neuron[i];j++){
            for (int k=0;k<num_neuron[i-1];k++){
                network->layers[i-1].neurons[k].dw[j] = network->layers[i].neurons[j].dz * network->layers[i-1].neurons[k].act;
                if (i>1){
                    network->layers[i-1].neurons[k].dz += ((network->layers[i].neurons[j].dz) * (network->layers[i-1].neurons[k].out_weights[j]) * sigmoid_prime(network->layers[i-1].neurons[k].z));
                }
            }
            network->layers[i].neurons[j].dbias = network->layers[i].neurons[j].dz;
        }

    }
}
//update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch
void update_mini_batch(dataset_t * mini_batch,Network * network)
{
    int i,j,k,l;
    Network* buffer = create_Network(); //store mean of dw, dbias.
    initialize_nabla(network);
    for (int i=0; i<mini_batch->size;i++){
        backprop(mini_batch,network,i);
        for (int j=0; j<num_layers;j++){
            for (int k=0; k<num_neuron[j];k++){
                if(j>0){
                    buffer->layers[j].neurons[k].dbias += (network->layers[j].neurons[k].dbias) / (float)(mini_batch->size);
                }
                if(j<num_layers-1){
                    for(int l=0;l<num_neuron[j+1];l++){
                        buffer->layers[j].neurons[k].dw[l] += (network->layers[j].neurons[k].dw[l]) / (float)(mini_batch->size);
                    }
                }
            }           
        }
    }
    //update the bias and weights by using sum of nabla_w, nable_b
    for (int i=0;i<num_layers;i++){
        for(int j=0;j<num_neuron[i];j++){
            if(i>0){
                network->layers[i].neurons[j].bias -= (eps * buffer->layers[i].neurons[j].dbias);
            }
            if(i<num_layers-1){
                for(int k=0;k<num_neuron[i+1];k++){
                    network->layers[i].neurons[j].out_weights[k] -= (eps * buffer->layers[i].neurons[j].dw[k]);
                }
            }
        }
    }


}
//Return the number of test inputs for which the neural network outputs the correct result.
//Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation
int compute_acc(Network * network,dataset_t * test_data){
    int label_index = 0;
    int index = 0;
    int correct = 0;
    test_data->size = 10;
    for (int i=0; i<test_data->size;i++){
        float max=0;
        feedforward(network,test_data,i);
        for (int j=0;j<num_neuron[num_layers-1];j++){
            if(network->layers[num_layers-1].neurons[j].act > max){
                max = network->layers[num_layers-1].neurons[j].act;
                index = j;
            }
            if(test_data->labels[i].desired_out[j] == 1){
                label_index = j;
            }
        }
        if( index ==label_index){
            correct= correct + 1;
        }
        
    }
    return correct;

}
//Train the neural network using mini-batch stochastic gradient descent.
void SGD(dataset_t * datasets,Network * network,dataset_t * test_data){
    int i,j,k;
    int acc;
    int mini = 0;
    dataset_t * batch = (dataset_t *)malloc(sizeof(dataset_t));
    for (int i=0; i<epoch;i++){
        for (int j=0; j<(DATASIZE)/batch_size;j++){
            mini = mini_batch(datasets,batch,batch_size,j);
            update_mini_batch(batch,network);
        }
        acc = compute_acc(network,test_data);
        printf("accuracy : %d (epoch : %d)\n",acc,i);
    }
}