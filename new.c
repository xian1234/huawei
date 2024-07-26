#include <arm_sme_draft_spec_subject_to_change.h>
#include <arm_sve.h>

#include "matrix_methods.h"


void MatrixMultiplySme(double *res, double *A, double *B, int m, int n, int k)
{
    __asm__ volatile("SMSTART SM" ::: "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");
    __asm__ volatile("SMSTART ZA" ::: "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");

    for(int p=0; p < k; p++) {

        for(int i = 0; i < m; i += svcntd()) {
            svbool_t pm = svwhilelt_b64(i, m);
            svint64_t vecI = svindex_s64(i * k + p, k);
            svfloat64_t vecA = svld1_gather_index(pm, &A[0], vecI);
            for(int j=0; j < n; j += svcntd()) {
                svbool_t pn = svwhilelt_b64(j, n);
                svfloat64_t vecB = svld1(pn, &B[p*n + j]);
                
                svzero_za();
                svmopa_za64_m(0, pm, pn, vecA, vecB);
                for(int l = 0; l < svcntd() && l < m - i; l++) {
                    svfloat64_t re1 = svld1(pn, &res[i * n + l *n + j]);
                    svfloat64_t re2;
                    re2 = svread_hor_za64_m(re2, pn, 0, l);
                    re2 = svadd_z(pn, re1, re2);

                    svst1(pn, &res[i * n + l *n + j], re2);
                }
            }
        }
    }

}
