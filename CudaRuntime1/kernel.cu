#include "kernel.cuh"

__host__ __device__
double get_value(int m, int n, int x_row, int y_row, double* list) {
	
	// m is the number of rows, n is the number of columns
	int index = x_row * n + y_row;

	return list[index];
}

__host__ __device__
int get_row_index(int m, int n, int index) {
	return index / n;
}

__host__ __device__
int get_col_index(int m, int n, int index) {
	return index % n;
}

__host__ __device__
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

__global__
void kernel(int matrix0_col, int n, int matrix1_row, double* matrix0, double* matrix1, double* c_output) {
	// 스레드 ID를 계산한다.
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	// blockDim.x * blockIdx.x는 블록의 첫 번째 스레드가 총 스레드중에 몇 번째 스레드인지를 나타낸다.
	// threadIdx.x는 블록 내에서 몇 번째 스레드인지를 나타낸다.
	// 이 둘을 더하면 총 스레드 중에서 몇 번째 스레드인지를 나타낸다.
	// 여기서 말하는 블록은 여러 개로 이루어진 스레드의 묶음을 의미한다.
	// 여러 스레드 블록을 모아 하나의 그리드를 만든다.

	// 스레드 ID가 행렬의 크기를 넘어가면 종료한다.
	if (id < matrix0_col * matrix1_row) {
		get_multiplied_matrix(matrix0_col, n, matrix1_row, id, matrix0, matrix1, c_output);
	}
}