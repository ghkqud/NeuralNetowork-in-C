/*********************************************
 * Program Name : get_data.c (get handmade 5X5 data)
 * 
 * Brief Description: A module to get 5X5 data.
 * 
 * Author : C.H.B
 * Created on : July 30, 2021.
 * Last modifed on: August 4, 2021. 
*/
#include <stdio.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include "smalldatafile.h"
#include <dirent.h>
#include <io.h>
#include <unistd.h>


#define _CRT_SECURE_NO_WARNINGS   
dataset_t get_file(int train,char * s_dir,char*s_label_dir)
{
    dataset_t small_data;
    int DATA_SIZE = 110;

    if(train == 0){
        DATA_SIZE = 10;
    }

    small_image_t * image = (small_image_t *)malloc(DATA_SIZE * sizeof(small_image_t));
    small_label_t * label = (small_label_t *)malloc(DATA_SIZE * sizeof(small_label_t));
    DIR *dir_info;
    struct dirent *dir_enrty;
    int count =0;
    
    if(train==0)
    {
        printf("test_data\n");
    }
    else
    {
        printf("train_data\n");

    }
    char * s_image_buffer = (char *)malloc(sizeof(char) *100);
    char * s_label_buffer = (char *)malloc(sizeof(char) *100);
    dir_info = opendir(s_dir);
    if (NULL !=dir_info){
        
        while((dir_enrty = readdir(dir_info)) !=NULL)
        {
            if(strcmp(dir_enrty->d_name, ".") == 0 || strcmp(dir_enrty->d_name, "..") == 0) 
            { 
                continue; 
            }

            //printf("%s\n",dir_enrty->d_name);
            float data[25];
            float buffer_f = 0;
            strcpy(s_image_buffer,s_dir);
            strcat(s_image_buffer,dir_enrty->d_name);
            strcpy(s_label_buffer,s_label_dir);
            strcat(s_label_buffer,dir_enrty->d_name);
            FILE * fp_image;
            FILE * fp_label;
            fp_image = fopen(s_image_buffer,"r");
            if(fp_image == NULL){
                printf("error");
            }
            fp_label = fopen(s_label_buffer,"r");
            int index =0;
            while (feof(fp_image) == 0)
            {
                fscanf(fp_image,"%f",&buffer_f);
                //printf("%f\n",buffer_f);
                image[count].pixels[index] = buffer_f;
                index ++;
                

            }
           
            index=0;
            while (feof(fp_label)==0)
            {
                fscanf(fp_label,"%f",&buffer_f);
                label[count].desired_out[index] = buffer_f;
                index ++;
            }
            
            count++;
            fclose(fp_image);
            fclose(fp_label);

        }
        closedir(dir_info);
    }
    small_data.images = image;
    small_data.labels = label;
    small_data.size = DATA_SIZE;
    return small_data;

}
