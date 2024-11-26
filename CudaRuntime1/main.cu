#include <stdio.h>
#include <stdlib.h>

#include "kernel.cuh"
#include <ctime>

int main(void) {
	cudaSetDevice(0);

	int matrix0_col = 300;
	int n = 100;
	int matrix1_row = 200;

	double* matrix0 = (double*)malloc(sizeof(double) * matrix0_col * n);
	double* matrix1 = (double*)malloc(sizeof(double) * n * matrix1_row);
	double* c_output = (double*)malloc(sizeof(double) * matrix0_col * matrix1_row);

	double* d_matrix0;
	double* d_matrix1;
	double* d_c_output;

	cudaMalloc(&d_matrix0, sizeof(double) * matrix0_col * n);
	cudaMalloc(&d_matrix1, sizeof(double) * n * matrix1_row);
	cudaMalloc(&d_c_output, sizeof(double) * matrix0_col * matrix1_row);

	for (int i = 0; i < matrix0_col * n; ++i) {
		matrix0[i] = 1.0;
	}

	for (int i = 0; i < n * matrix1_row; ++i) {
		matrix1[i] = 1.0;
	}

	for (int i = 0; i < matrix0_col * matrix1_row; ++i) {
		c_output[i] = 0.0;
	}

	cudaMemcpy(d_matrix0, matrix0, sizeof(double) * matrix0_col * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix1, matrix1, sizeof(double) * n * matrix1_row, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c_output, c_output, sizeof(double) * matrix0_col * matrix1_row, cudaMemcpyHostToDevice);

	cudaDeviceProp prop; // ����̽��� ������Ƽ�� ������ ����ü
	cudaGetDeviceProperties(&prop, 0); // 0�� ����̽��� ������ ����ü�� ����
	int n_threads = prop.maxThreadsPerBlock; // ��ϴ� �ִ� ������ ����. �ִ� �� ���� ����� �� �ִ��� Ȯ�ο�
	// �ִ�� ����ϳ� ���� �ھ� ���� ����ϵ� ���̰� ���� ����.

	int n_blocks = prop.multiProcessorCount; // ��Ƽ ���μ����� ����. ��Ƽ ���μ����� ����� �ϳ��� �Ҵ��Ѵ�.

	// GPU computing
	cudaEvent_t d_start, d_stop;

	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	cudaEventRecord(d_start);

	// <<<B, T>>> B�� ����� ����, T�� ��ϴ� �������� ����
	kernel <<<n_blocks, n_threads >>> (matrix0_col, n, matrix1_row, d_matrix0, d_matrix1, d_c_output);

	cudaEventRecord(d_stop);

	cudaMemcpy(c_output, d_c_output, sizeof(double) * matrix0_col * matrix1_row, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(d_stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d_start, d_stop);

	printf("GPU time: %f ms\n", elapsedTime);

	// CPU computing
	double* cpu_c_output = (double*)malloc(sizeof(double) * matrix0_col * matrix1_row);
	
	for (int i = 0; i < matrix0_col * matrix1_row; ++i) {
		cpu_c_output[i] = 0.0;
	}
	
	clock_t start = clock();
	clock_t diff;

	for (int i = 0; i < matrix0_col * matrix1_row; ++i) {
		get_multiplied_matrix(matrix0_col, n, matrix1_row, i, matrix0, matrix1, cpu_c_output);
	}

	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;

	printf("CPU time: %d ms\n", msec);

	// print result
	for (int i = 0; i < matrix1_row; ++i) {
		for (int j = 0; j < matrix0_col; ++j) {
			printf("%f ", c_output[i * matrix0_col + j]);
		}
		printf("\n");
	}

	cudaFree(d_matrix0);
	cudaFree(d_matrix1);
	cudaFree(d_c_output);

	free(matrix0);
	free(matrix1);
	free(c_output);
	free(cpu_c_output);
}