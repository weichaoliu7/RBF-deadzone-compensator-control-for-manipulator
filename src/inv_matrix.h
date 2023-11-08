#ifndef INV_MATRIX_H
#define INV_MATRIX_H

#include <stdlib.h>

void inv_matrix(double inv_matrix[][2], double matrix[][2], int n) {
    double identity[n][n];
    double matrix_p[n][n];
    double ratio, temp;
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j)
                identity[i][j] = 1;
            else
                identity[i][j] = 0;

            matrix_p[i][j] = matrix[i][j];
        }
    }

    for (i = 0; i < n; i++) {
        temp = matrix_p[i][i];
        for (j = 0; j < n; j++) {
            matrix_p[i][j] /= temp;
            identity[i][j] /= temp;
        }

        for (j = 0; j < n; j++) {
            if (j != i) {
                ratio = matrix_p[j][i];
                for (k = 0; k < n; k++) {
                    matrix_p[j][k] -= ratio * matrix_p[i][k];
                    identity[j][k] -= ratio * identity[i][k];
                }
            }
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            inv_matrix[i][j] = identity[i][j];
        }
    }
}

#endif
