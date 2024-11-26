#include <stdio.h>
#include <stdlib.h>

#include "kernel.cuh"

int main(void) {

	int matrix0_col = 3;
	int n = 3;
	int matrix1_row = 1;

	double* matrix0 = (double*)malloc(sizeof(double) * matrix0_col * n);
	double* matrix1 = (double*)malloc(sizeof(double) * n * matrix1_row);
	double* c_output = (double*)malloc(sizeof(double) * matrix0_col * matrix1_row);

	for (int i = 0; i < matrix0_col * n; ++i) {
		matrix0[i] = 1.0;
	}

	for (int i = 0; i < n * matrix1_row; ++i) {
		matrix1[i] = 1.0;
	}

	for (int i = 0; i < matrix0_col * matrix1_row; ++i) {
		c_output[i] = 0.0;
	}

	kernel(matrix0_col, n, matrix1_row, matrix0, matrix1, c_output);

	for (int i = 0; i < matrix1_row; ++i) {
		for (int j = 0; j < matrix0_col; ++j) {
			printf("%f ", c_output[i * matrix0_col + j]);
		}
		printf("\n");
	}
}