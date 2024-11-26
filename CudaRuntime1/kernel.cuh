#ifndef KERNEL_CUH
#define KERNEL_CUH


double get_value(int m, int n, int x_row, int y_row, double* list);

int get_row_index(int m, int n, int index);

int get_col_index(int m, int n, int index);

void get_multiplied_matrix(int matrix0_col, int n, int matrix1_row, int index, double* matrix0, double* matrix1, double* c_output);

void kernel(int matrix0_col, int n, int matrix1_row, double* matrix0, double* matrix1, double* c_output);

#endif /* KERNEL_CUH */