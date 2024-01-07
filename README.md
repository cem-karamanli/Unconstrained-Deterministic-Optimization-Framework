A framework for solving the unconstrained deterministic optimization problems.

The problems considered are logistic regression, rosenbrock and quadratic problems. 

The implemented methods are SD (steepest descent) method, Newton's method, BFGS_Bk method (BFGS method which uses B_k matrices in the update), BFGS_Hk method (BFGS method which uses H_k matrices in the update) and LBFGS method.

For the quadratic problems, a prespecified condition number (kappa) is used.
Armijo line-search is implemented as an option to further increase the performance of the algorithms.
Various datasets for logistic regression problems can be obtained from the LIBSVM collection at: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
