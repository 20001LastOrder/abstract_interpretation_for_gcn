# Abstract Interpretation for Robustness Certification of Graph Convolutional Networks
Additional details for the paper: Certifying Robustness of Graph Convolutional Networks for Node Perturbation with Polyhedra Abstract Interpretation

Further details including:
* Implementation of Poly and Interval methods in the paper
* Reproducibility package of experiments in the paper

## Project setup
1. Run `prepare.sh` to get data and the Dual method used in the paper
2. Run `setup.sh` to install required libraries
3. `experiment_scripts` folder contains scripts for the lowerbound, runtime experiment, and scripts to calculate input for the collective certification 
    * `robust-gcn` contains implementation for the Dual method of the paper
    * `abstract_interpretation/interval` contains implementation for the interval method
    * `abstract_interpretation/node_deeppoly` contains implementation for the poly method
4. run `experiments.py` for the robust training
5. the folder `collective_certification/` contains implementation for the collective certification method adapted from Schuchardt et al (https://github.com/jan-schuchardt/collective_robustness). Please follow the Readme in the folder for more details.

### Mapping of approaches to the paper
* interval -> Interval
* poly -> Poly-TopK
* poly_max -> Poly-Max
* optim -> Dual-F
* optim_origin -> Dual-O

### A note on Dual-F
The original implementation of Dual (Dual-O) contains an implementation issue due to the sparse matrix multiplication, making the upper-bound estimation higher than the actual value. To correct this (run Dual-F), change line 123 under `robust-gcn/robust_gcn/robust_gcn.py` to 

``` python
return adj_slice.to_dense().mm(input.mm(self.weights)) + self.bias
```