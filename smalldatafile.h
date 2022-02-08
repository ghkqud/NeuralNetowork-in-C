#ifndef DATA_FILE_H
#define DATA_FILE_H


typedef struct small_image_t_{
    float pixels[25];
    int num_image;
} small_image_t;
typedef struct small_label_t_{
    int desired_out[10];
    int num_label;
} small_label_t;
typedef struct dataset_t_{
    small_image_t * images;
    small_label_t * labels;
    int size;
} dataset_t;

dataset_t get_file(int train,char * s_dir,char*s_label_dir);
#endif