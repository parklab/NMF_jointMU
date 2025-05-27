This repository contains a Python implementation of the **joint multiplicative update (MU) algorithm** for non-negative matrix factorization (NMF).

The joint MU algorithm minimizes the Kullback-Leibler (KL) divergence by simultaneously updating both matrices of the decomposition, in contrast to the classical MU algorithm which alternates updates for one matrix at a time. This joint update avoids a matrix multiplication in each iteration, resulting in a runtime improvement of approximately 40%.
