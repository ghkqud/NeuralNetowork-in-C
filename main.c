/***************************************************
 * Program Name : main.c (This C code is modified from Network.py)
 * Brief Description : A module to implement the stochastic gradient descent learning
 *      algorithm for a feedforward neural network by using 5 X 5 small data. 
 *      Gradients are calculated using backpropagation module that is saved in network.c file.
 *      5 X 5 small data get from get_data.c file 
 * 
 * Usage Instructions : Execute "g++ main.c network.c get_data.c -lm -o mnist".
 *      after that, Excute "./mnist"
 * 
 * Author : C.H.B
 * Created on: July 30, 2021.
 * Last modifed on: August 5, 2021.
 * 
 */

// Include files - include the basic set of definitions and functions
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <io.h>
#include <unistd.h>
#include <time.h>
#define eps 0.01
#define epoch 1000
#define batch_size 10
#define num_layers 3
#define PI 3.1415926535897932384
#include <time.h>

// Include files - include network module and get 5X5 data module
#include "smalldatafile.h"
#include "network.h"

// Datasize is used for calculating Batch index
#define DATASIZE 10


char * s_dir_train = "C:\\Users\\esa\\Desktop\\small_data_train\\image_train\\";
char * s_label_dir_train = "C:\\Users\\esa\\Desktop\\small_data_train\\label_train\\";
char * s_dir_test = "C:\\Users\\esa\\Desktop\\small_data_test\\image_test\\";
char * s_label_dir_test = "C:\\Users\\esa\\Desktop\\small_data_test\\label_test\\";


//function for normalization distribution (this code not use)
/*float normal(float stddev, float mean)
{
    float r1, r2, z1, z2, result;
    int i;

    result =0.0;
    r1 = ((float)rand()+1.) / ((float)RAND_MAX+1.);
    r2 = ((float)rand()+1.) / ((float)RAND_MAX+1.);

    z1 = sqrt(-2 * log(r1)) * cos(2 * PI * r2);
    result = z1*stddev + mean;
 
    return result;
}*/

// main code to implement neural network for 5 X 5 small data 
int main(){
    clock_t start,end; // for calculating elabsed time
    
    // declare training_data, test_data and allocate address of data
    dataset_t* training_data = (dataset_t *)malloc(sizeof(dataset_t));
    dataset_t*  test_data = (dataset_t *)malloc(sizeof(dataset_t));
    training_data->images = (small_image_t *)malloc(DATASIZE * sizeof(small_image_t));
    training_data->labels = (small_label_t *)malloc(DATASIZE * sizeof(small_label_t));
    test_data->images = (small_image_t *)malloc(DATASIZE * sizeof(small_image_t));
    test_data->labels = (small_label_t *)malloc(DATASIZE * sizeof(small_label_t));
    // declare network, network consists of layers that consists of neurons.
    Network* network = create_Network();

    *training_data= get_file(1,s_dir_train,s_label_dir_train);
    *test_data = get_file(0,s_dir_test,s_label_dir_test);
    //Implement Neural Network by using Stochastic Gradient Descent
    /*for (int i=0; i< 110;i++){
        for (int j=0;j<10;j++){
            printf(" %2f",training_data[0].labels[j]);
        }
        printf("\n");
    
    }*/
    start = clock();
    SGD(training_data,network,test_data);
    end = clock();
    printf(" \nelabsed time : %.3f",(float)(end-start)/CLOCKS_PER_SEC);
    free(network->layers);
    free(training_data->images);
    free(test_data->images);
    free(training_data->labels);
    free(test_data->labels);
    return 0;
}