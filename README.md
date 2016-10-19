# cumf_sgd

##Introduction
cumf_sgd is a CUDA-based SGD solution for large-scale MF problems. It uses batch-Hogwild! paralelize the tasks. It also has highly-optimized kernels for SGD update, leveraging cache, warp-shuffle instructions and half-precision floats. 

On three data sets (Netflix, Yahoo!Music and Hugewiki) with one Maxwell or Pascal GPU, cumf_sgd runs 3.1X-28.2X as fast compared with state-of-art CPU solutions on 1-64 CPU nodes. 

