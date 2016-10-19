#ifndef _SGD_GPU_H_
#define _SGD_GPU_H_

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace std;

typedef int SGDIndex; 
typedef float SGDRate;

SGDIndex const kALIGNByte = 4;
SGDIndex const kALIGN = kALIGNByte/sizeof(SGDRate);

struct Parameter
{
    int k;
    int num_blocks;
    int u_grid;
    int v_grid;
    int x_grid;
    int y_grid;
    int ux;
    int vy;
    int num_iters;
    int gpu;
   	SGDRate lambda_p;
    SGDRate lambda_q;
    SGDRate lrate;
    float alpha,beta;
    bool do_nmf;
    bool quiet;
    bool copy_data;
    Parameter():k(80), num_blocks(12), u_grid(1), v_grid(1), x_grid(1), y_grid(1), ux(1), vy(1), num_iters(30), gpu(3), lambda_p(0.05), lambda_q(0.05), lrate(0), alpha(0.01), beta(0.01){}
};

struct Argument
{
    Argument() : param(), on_disk(false), do_cv(false) {}
    string tr_path, va_path, model_path;
    Parameter param;
    bool on_disk;
    bool do_cv;

    void print_arg()
    {
    	printf("k         :%d\n",  param.k);
    	printf("num_blocks:%d\n",  param.num_blocks);
        printf("u_grid    :%d\n",  param.u_grid);
        printf("v_grid    :%d\n",  param.v_grid);
        printf("x_grid    :%d\n",  param.x_grid);
        printf("y_grid    :%d\n",  param.y_grid);
        printf("ux        :%d\n",  param.ux);
        printf("vy        :%d\n",  param.vy);
    	printf("num_iters :%d\n",  param.num_iters);
        printf("gpu       :%d\n",  param.gpu);
    	printf("lambda_p  :%.5f\n",param.lambda_p);
    	printf("lambda_q  :%.5f\n",param.lambda_q);
    	printf("lrate     :%.5f\n",param.lrate);
        printf("alpha     :%.4f\n",param.alpha);
        printf("beta      :%.4f\n",param.beta);
    	printf("tr_path   :%s\n",  tr_path.c_str());
    	printf("va_path   :%s\n",  va_path.c_str());
    	printf("model_path:%s\n",  model_path.c_str());
    }
};

struct mf_node
{
	SGDIndex u,v;
	SGDRate rate;
};

struct mf_problem
{
    SGDIndex m;
    SGDIndex n;
    long long nnz;
    int u_grid, v_grid;
    int x_grid, y_grid;
    int ux, vy;
    long long u_seg, v_seg;

    struct mf_node *R;
    struct mf_node **R2D;
    long long *gridSize;
    long long maxGridSize;

    struct mf_node *gpuR;
    int cur_u_id;
    int cur_v_id;

    struct mf_node *gpuRptrs[2];
    int cur_global_x_id[2];
    int cur_global_y_id[2];

};

struct mf_model
{
    int fun;
    SGDIndex m;
    SGDIndex n;
    int u_grid, v_grid;
    int x_grid, y_grid;
    int ux, vy;
    long long u_seg, v_seg;

    int k;
    float b;
    float *floatp;
    float *floatq;
    short *halfp;
    short *halfq;

    half *gpuHalfp;
    half *gpuHalfq;
    int cur_u_id;
    int cur_v_id;

    half *gpuHalfPptrs[2];
    half *gpuHalfQptrs[2];

    int cur_global_x_id[2];
    int cur_global_y_id[2];
    
};

mf_problem read_problem(string path);
mf_model* sgd_train(mf_problem*, mf_problem*, Parameter);

#endif


