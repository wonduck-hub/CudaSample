#include "kernel.cuh"

double get_value(int m, int n, int x_row, int y_row, double* list) {
	// m is the number of rows, n is the number of columns
	int index = x_row * n + y_row;

	return list[index];
}

int get_row_index(int m, int n, int index) {
	return index / n;
}

int get_col_index(int m, int n, int index) {
	return index % n;
}

void get_multiplied_matrix(int matrix0_col, int n, int matrix1_row, int index, double* matrix0, double* matrix1, double* c_output) {
	c_output[index] = 0.0;

	// c_output의 행렬은 m x k 크기다
	// 행렬곱을 하기 위해서는 matrix0의 열의 수(n)와 matrix1의 행의 수(n)가 같아야 한다.	
	// n을 공유해 사용하는 것
	int x_row = get_row_index(matrix0_col, matrix1_row, index);
	int y_row = get_col_index(matrix0_col, matrix1_row, index);

	for (int i = 0; i < n; ++i) {
		double temp0 = get_value(matrix0_col, n, x_row, i, matrix0);
		double temp1 = get_value(n, matrix1_row, i, y_row, matrix1);

		c_output[index] += temp0 * temp1;
	}
}

void kernel(int matrix0_col, int n, int matrix1_row, double* matrix0, double* matrix1, double* c_output) {
	for (int index = 0; index < matrix0_col * matrix1_row; ++index) {
		get_multiplied_matrix(matrix0_col, n, matrix1_row, index, matrix0, matrix1, c_output);
	}
}