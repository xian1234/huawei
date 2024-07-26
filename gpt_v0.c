#include <arm_sve.h>
#include <iostream>

#define A(i,j) a[ (j)*n + (i) ]
#define B(i,j) b[ (j)*n + (i) ]
#define C(i,j) c[ (j)*n + (i) ]

//computing (4xk)x(kx4) dot product using SVE
void add_dot_4x4_sve(int n, int k, double* a, double* b, double* c) {
    double *b_ptr_0, *b_ptr_1, *b_ptr_2, *b_ptr_3;

    b_ptr_0 = &B(0, 0);
    b_ptr_1 = &B(0, 1);
    b_ptr_2 = &B(0, 2);
    b_ptr_3 = &B(0, 3);

    svfloat64_t c_sum_0 = svdup_f64(0.0);
    svfloat64_t c_sum_1 = svdup_f64(0.0);
    svfloat64_t c_sum_2 = svdup_f64(0.0);
    svfloat64_t c_sum_3 = svdup_f64(0.0);

    for (int p = 0; p < k; p++) {
        svfloat64_t a_reg = svld1_f64(svptrue_b64(), &A(0, p));

        double b_reg_0 = *(b_ptr_0++);
        double b_reg_1 = *(b_ptr_1++);
        double b_reg_2 = *(b_ptr_2++);
        double b_reg_3 = *(b_ptr_3++);

        c_sum_0 = svmla_f64(c_sum_0, a_reg, svdup_f64(b_reg_0));
        c_sum_1 = svmla_f64(c_sum_1, a_reg, svdup_f64(b_reg_1));
        c_sum_2 = svmla_f64(c_sum_2, a_reg, svdup_f64(b_reg_2));
        c_sum_3 = svmla_f64(c_sum_3, a_reg, svdup_f64(b_reg_3));
    }

    double *c_ptr = 0;
    c_ptr = &C(0, 0);
    svfloat64_t c_reg = svld1_f64(svptrue_b64(), c_ptr);
    c_reg = svadd_f64(c_reg, c_sum_0);
    svst1_f64(svptrue_b64(), c_ptr, c_reg);

    c_ptr = &C(0, 1);
    c_reg = svld1_f64(svptrue_b64(), c_ptr);
    c_reg = svadd_f64(c_reg, c_sum_1);
    svst1_f64(svptrue_b64(), c_ptr, c_reg);

    c_ptr = &C(0, 2);
    c_reg = svld1_f64(svptrue_b64(), c_ptr);
    c_reg = svadd_f64(c_reg, c_sum_2);
    svst1_f64(svptrue_b64(), c_ptr, c_reg);

    c_ptr = &C(0, 3);
    c_reg = svld1_f64(svptrue_b64(), c_ptr);
    c_reg = svadd_f64(c_reg, c_sum_3);
    svst1_f64(svptrue_b64(), c_ptr, c_reg);
}

void solution_2_sve(int n, double* a, double* b, double* c) {
    int i, j, k;
    for (j = 0; j < ((n) & (~3)); j += 4) {
        for (i = 0; i < ((n) & (~3)); i += 4) {
            add_dot_4x4_sve(n, n, &A(i, 0), &B(0, j), &C(i, j));
        }
        for (; i < n; i++) {
            double c_0, c_1, c_2, c_3;
            c_0 = C(i, j);
            c_1 = C(i, j + 1);
            c_2 = C(i, j + 2);
            c_3 = C(i, j + 3);
            for (int k = 0; k < n; k++) {
                c_0 += A(i, k) * B(k, j);
                c_1 += A(i, k) * B(k, j + 1);    
                c_2 += A(i, k) * B(k, j + 2);    
                c_3 += A(i, k) * B(k, j + 3);  
            }
            C(i, j) = c_0;
            C(i, j + 1) = c_1;
            C(i, j + 2) = c_2;
            C(i, j + 3) = c_3;
        }
    }  
    for (; j < n; j++) {
        for (i = 0; i < ((n) & (~3)); i += 4) {
            svfloat64_t buf = svld1_f64(svptrue_b64(), &C(i, j));
            for (int k = 0; k < n; k++) {
                svfloat64_t va = svld1_f64(svptrue_b64(), &A(i, k));
                double vb = B(k, j);
                buf = svmla_f64(buf, va, svdup_f64(vb));
            }
            svst1_f64(svptrue_b64(), &C(i, j), buf);
        }
        for (; i < n; i++) {
            double temp = C(i, j);
            for (int k = 0; k < n; k++) {
                temp += A(i, k) * B(k, j);
            }
            C(i, j) = temp;
        }
    }  
}
