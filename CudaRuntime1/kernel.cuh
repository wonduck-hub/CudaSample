#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <omp.h> // OpenMP
#include <cuda.h>

__host__ __device__
double get_value(int m, int n, int x_row, int y_row, double* list);

__host__ __device__
int get_row_index(int m, int n, int index);

__host__ __device__
int get_col_index(int m, int n, int index);

__host__ __device__
void get_multiplied_matrix(int matrix0_col, int n, int matrix1_row, int index, double* matrix0, double* matrix1, double* c_output);

__global__
void kernel(int matrix0_col, int n, int matrix1_row, double* matrix0, double* matrix1, double* c_output);

#endif /* KERNEL_CUH */