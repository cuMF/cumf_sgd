# CuMF_sgd

##Introduction

Matrix factorization has been demonstrated to be effective in recommender system, topic modeling, word embedding, and other machine learning applications. As the input data set is often large, MF solution are time-consuming. Therefore, how to solve MF problems efficiently is an important problem. There are mainly three algorithms to solve MF, coordinate gradient descent(CGD), alternate least square(ALS), and stochastic gradient descent(SGD). Our previous project tackles ALS acceleration on GPUs, we foucs on SGD solution in this project and present **cuMF_SGD**.


<img src=https://github.com/CuMF/cumf_sgd/raw/master/figures/mf.png width=405 height=161 />


CuMF_SGD is a CUDA-based SGD solution for large-scale matrix factorization(MF) problems. CuMF_SGD is able to solve MF problems with one or multiple GPUs within one single node. It first partitions the input data into matrix blocks and distribute them to different GPUs. Then it uses batch-Hogwild! algorithm to parallelize the SGD updates withn one GPU. It also has highly-optimized kernels for SGD update, leveraging cache, warp-shuffle instructions, and half-precision floats.



We test cuMF_SGD using three data sets (Netflix, Yahoo!Music and Hugewiki) with one Maxwell or Pascal GPU, cumf_sgd runs 3.1X-28.2X as fast compared with state-of-art CPU solutions on 1-64 CPU nodes. We also test Yahoo!Music on two Pascal GPUs and we observer that two GPUs provides ~30% speedup over one GPU. 

Note: the repository only contains single GPU version, the multiple GPU version still needs more test.

## Compilation 

Run the Makefile in the source code directory.

## Input data format

The input rating is organized as follows:
    user_id item_id rating
user_id and item_id are 4-byte integers and rating is 4-byte floating point. They are all stored in binary format. 


To facilitate your development, we describe how to prepare input data sets and use Netflix data set as example. The Netflix data sets can be downloaded from [CMU Graphlab](http://www.select.cs.cmu.edu/code/graphlab/datasets/). 

We use netflix_mm/netflix_me as the train/test sets. The orginal files are in text format, not binary format. You can transform the format using the script located in data/netflix/run.sh. Download [netflix_mm](http://www.select.cs.cmu.edu/code/graphlab/datasets/netflix_mm) and [netflix_mme](http://www.select.cs.cmu.edu/code/graphlab/datasets/netflix_mme) , put the files in ./data/netflix run the script:
	./data/netflix/run.sh
Then you can use netflix_mm.bin as the train set and netflix_mme.bin as the test set. 



CuMF_SGD does not include a script to run the test, we suggest you use the [Libmf](https://github.com/cjlin1/libmf) to run the test.

## Run


## Reference

## Existing Issues

Details can be found at:

Xiaolong Xie, [Wei Tan](https://github.com/wei-tan), [Liana Fong](https://github.com/llfong), Yun Liang, CuMF_SGD: Fast and Scalable Matrix Factorization, arXiv preprint, arXiv:1610.05838([link](https://arxiv.org/abs/1610.05838)).

Our ALS-based MF solution can be found here:

Faster and Cheaper: Parallelizing Large-Scale Matrix Factorization on GPUs. [Wei Tan](https://github.com/wei-tan), [Liangliang Cao](https://github.com/llcao), [Liana Fong](https://github.com/llfong). [HPDC 2016], Kyoto, Japan. [(arXiv)](http://arxiv.org/abs/1603.03820) [(github)](https://github.com/wei-tan/cumf_als)

