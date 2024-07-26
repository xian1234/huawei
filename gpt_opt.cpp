#include <arm_sve.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define A(i,j) A[ (j)*m + (i) ]
#define B(i,j) B[ (j)*k + (i) ]
#define C(i,j) res[ (j)*m + (i) ]

void add_dot_4x4_sve(int m, int k, double* a, double* b, double* c) {
    svbool_t pg = svptrue_b64();
    svfloat64_t c_sum_0 = svdup_f64(0.0);
    svfloat64_t c_sum_1 = svdup_f64(0.0);
    svfloat64_t c_sum_2 = svdup_f64(0.0);
    svfloat64_t c_sum_3 = svdup_f64(0.0);

    for (int p = 0; p < k; p++) {
        svfloat64_t a_reg = svld1(pg, &A(0, p));

        double b_reg_0 = B(p, 0);
        double b_reg_1 = B(p, 1);
        double b_reg_2 = B(p, 2);
        double b_reg_3 = B(p, 3);

        c_sum_0 = svmla_f64(c_sum_0, a_reg, svdup_f64(b_reg_0));
        c_sum_1 = svmla_f64(c_sum_1, a_reg, svdup_f64(b_reg_1));
        c_sum_2 = svmla_f64(c_sum_2, a_reg, svdup_f64(b_reg_2));
        c_sum_3 = svmla_f64(c_sum_3, a_reg, svdup_f64(b_reg_3));
    }

    svfloat64_t c_reg = svld1(pg, &C(0, 0));
    c_reg = svadd_f64(c_reg, c_sum_0);
    svst1(pg, &C(0, 0), c_reg);

    c_reg = svld1(pg, &C(0, 1));
    c_reg = svadd_f64(c_reg, c_sum_1);
    svst1(pg, &C(0, 1), c_reg);

    c_reg = svld1(pg, &C(0, 2));
    c_reg = svadd_f64(c_reg, c_sum_2);
    svst1(pg, &C(0, 2), c_reg);

    c_reg = svld1(pg, &C(0, 3));
    c_reg = svadd_f64(c_reg, c_sum_3);
    svst1(pg, &C(0, 3), c_reg);
}

void MatrixMultiplySme(double *res, double *A, double *B, int m, int n, int k) {
    int i, j, l;
    for (j = 0; j < ((n) & (~3)); j += 4) {
        for (i = 0; i < ((m) & (~3)); i += 4) {
            add_dot_4x4_sve(m, k, &A(i, 0), &B(0, j), &C(i, j));
        }
        for (; i < m; i++) {
            double c_0 = C(i, j);
            double c_1 = C(i, j + 1);
            double c_2 = C(i, j + 2);
            double c_3 = C(i, j + 3);
            for (l = 0; l < k; l++) {
                c_0 += A(i, l) * B(l, j);
                c_1 += A(i, l) * B(l, j + 1);    
                c_2 += A(i, l) * B(l, j + 2);    
                c_3 += A(i, l) * B(l, j + 3);  
            }
            C(i, j) = c_0;
            C(i, j + 1) = c_1;
            C(i, j + 2) = c_2;
            C(i, j + 3) = c_3;
        }
    }  
    for (; j < n; j++) {
        for (i = 0; i < ((m) & (~3)); i += 4) {
            svfloat64_t buf = svld1(svptrue_b64(), &C(i, j));
            for (l = 0; l < k; l++) {
                svfloat64_t va = svld1(svptrue_b64(), &A(i, l));
                double vb = B(l, j);
                buf = svmla_f64(buf, va, svdup_f64(vb));
            }
            svst1(svptrue_b64(), &C(i, j), buf);
        }
        for (; i < m; i++) {
            double temp = C(i, j);
            for (l = 0; l < k; l++) {
                temp += A(i, l) * B(l, j);
            }
            C(i, j) = temp;
        }
    }  
}

// Helper function to initialize matrices with random values
void initialize_matrices(double *A, double *B, double *C, int m, int n, int k) {
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < m * k; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < k * n; ++i) {
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < m * n; ++i) {
        C[i] = 0.0;
    }
}

int main() {
    int m = 512, n = 512, k = 512;

    std::vector<double> A(m * k);
    std::vector<double> B(k * n);
    std::vector<double> C(m * n);

    initialize_matrices(A.data(), B.data(), C.data(), m, n, k);

    MatrixMultiplySme(C.data(), A.data(), B.data(), m, n, k);

    std::cout << "Matrix multiplication completed." << std::endl;
    return 0;
}
