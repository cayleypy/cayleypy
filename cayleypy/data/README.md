Pre-computed results:
* `*_cayley_growth.csv` - growth function for various Cayley graphs. 
    Key is size of permutations. 
    Central state doesn't matter (but assumed to be the identity permutation).
* `*_coset_growth.csv` - growth functions for various Schreier coset graphs. 
    Key is the central state.
* `puzzles_growth.csv` - growth functions for puzzles (when it can be fully computed).
* `heisenberg_growth.csv` - growth function for Cayley group for Heisenberg group (with inverses). 
    Key `n,modulo` where `n` is the size of the matrix and `modulo` is the modulo 
    under which matrix multiplication is done.
* `sl_fund_roots_n_growth.csv` - growth function for fundamental roots of SL(n) (n=2,3). Key is modulo.
* `sl_n_root_weyl.csv` - growth function os SL(n) (n=2,3) w.r.t. a root element and Coxeter element. Key is modulo.
