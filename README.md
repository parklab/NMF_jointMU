This repository contains a Python implementation of the **joint multiplicative updates (MU) algorithm** for non-negative matrix factorization (NMF) with the Kullback-Leibler (KL) divergence.

The joint MU algorithm minimizes the objective function by simultaneously updating both matrices of the decomposition, in contrast to the classical MU algorithm which alternates updates for one matrix at a time. The joint update avoids a matrix multiplication in each iteration, resulting in a runtime improvement of approximately 40%.
