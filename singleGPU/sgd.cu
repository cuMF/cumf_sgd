
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
#include <tuple>

#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <curand.h>
#include <curand_kernel.h>

#include <sstream>


#include <unistd.h>
#include <math.h>

#if defined USEOMP
#include <omp.h>
#endif

#include "sgd.h"

#include "sgd_kernel.h"

#include <iomanip>
using namespace std;

SGDIndex* gen_random_map(SGDIndex size)
{
    //srand(0);
    srand (time(NULL));
    vector<SGDIndex> map(size, 0);
    for(SGDIndex i = 0; i < size; i++)
        map[i] = i;

    random_shuffle(map.begin(), map.end());
    
    int*map_ptr = new int[size];
    for(int i = 0;i < size;i++)map_ptr[i] = map[i];
    
    return map_ptr;
}

SGDIndex* gen_inv_map(SGDIndex*map,int size)
{
    int*inv_map = new int[size];
    for(int i = 0;i < size;i++)inv_map[map[i]] = i;
    return inv_map;
}

struct sort_node_by_p
{
    bool operator() (mf_node const &lhs, mf_node const& rhs)
    {
        return tie(lhs.u, lhs.v) < tie(rhs.u, rhs.v);
    }
};

struct sort_node_by_q
{
    bool operator() (mf_node const &lhs, mf_node const &rhs)
    {
        return tie(lhs.v, lhs.u) < tie(rhs.v, rhs.u);
    }
};

void collect_data(mf_problem *prob, SGDRate& ave, SGDRate& std_dev)
{
    double ex = 0;
    double ex2 = 0;
    
    for(long long i = 0; i < prob->nnz; i++)
    {
        SGDRate r = prob->R[i].rate;
        ex += (double)r;
        ex2 += (double)r*r;
    }
    ex  = ex/(double)prob->nnz;
    ex2 = ex2/(double)prob->nnz;
    


    ave = (SGDRate)ex;
    std_dev = (SGDRate)sqrt(ex2-ex*ex);
}

void scale_problem(mf_problem*prob, float scale, long long u_seg, long long v_seg)
{

    if(prob->ux*prob->vy == 1)
    {
        for(long long i = 0;i < prob->nnz; i++)
        {   
            prob->R[i].rate = prob->R[i].rate*scale;
        }
    }
    else
    {

        for(long long i = 0;i < prob->nnz; i++)
        {   
            prob->R[i].rate = prob->R[i].rate*scale;

            long long tmp_u = prob->R[i].u;
            while(tmp_u >= u_seg)tmp_u = tmp_u - u_seg;
            prob->R[i].u = tmp_u;

            long long tmp_v = prob->R[i].v;
            while(tmp_v >= v_seg)tmp_v = tmp_v - v_seg;
            prob->R[i].v = tmp_v;
        }
    }
}

void shuffle_problem(mf_problem*prob, SGDIndex*p_map, SGDIndex*q_map)
{
    for(long long i = 0; i < prob->nnz; i++)
    {
        mf_node &N = prob->R[i];
        N.u = p_map[N.u];
        N.v = q_map[N.v];
    }
}


struct pthread_arg
{
    int thread_id; 
    string path;
    mf_node *R;
    long long offset;
    long long size;
    int max_m;
    int max_n;

};

void *read_problem_thread(void *argument)
{
    pthread_arg *arg = (pthread_arg*)argument;

    FILE*fptr = fopen(arg->path.c_str(), "rb");
    if(fptr == NULL)
    {
        printf("file %s open failed\n", arg->path.c_str());
        exit(0);
    }

    int max_m = -1;
    int max_n = -1;
    
    for(long long idx = 0;idx < arg->size;idx ++)
    {
        int flag = 0;
        int u,v;
        float r;

        flag += fread(&u, sizeof(int), 1, fptr); 
        flag += fread(&v, sizeof(int), 1, fptr); 
        flag += fread(&r, sizeof(float), 1, fptr); 

        if(flag != 3)break;

        if(u + 1 > max_m)max_m = u + 1;
        if(v + 1 > max_n)max_n = v + 1;

        arg->R[idx + arg->offset].u = u;
        arg->R[idx + arg->offset].v = v;
        arg->R[idx + arg->offset].rate = r;

    }
    fclose(fptr);

    arg->max_m = max_m;
    arg->max_n = max_n;
    return NULL;
}

mf_problem read_problem(string path)
{
    clock_t start;
    printf("read_problem called\n");

    start = clock();
    struct timespec begin, end;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &begin);

    mf_problem prob;
    prob.m = 1;
    prob.n = 1;
    prob.nnz = 0;
    prob.R = NULL;

    int num_files = 0;
    vector<string> file_names;
    for(int i = 0; i < 80;i++)
    {
        stringstream tmp_name_stream;
        tmp_name_stream << path << i;
        string tmp_name = tmp_name_stream.str();

        if(access(tmp_name.c_str(), F_OK) != -1)file_names.push_back(tmp_name);
    }
    num_files = file_names.size();


    if(num_files <= 0)
    {
        if(path.empty())
        {
            printf("file %s open failed\n", path.c_str());
            exit(0);
            return prob;
        }

        FILE*fptr = fopen(path.c_str(), "rb");
        if(fptr == NULL)
        {
            printf("file %s open failed\n", path.c_str());
            exit(0);
            return prob;
        }
        fseek(fptr, 0L, SEEK_END);
        prob.nnz = ftell(fptr)/12;

        mf_node *R;
        cudaMallocHost(&R,sizeof(mf_node)*prob.nnz); 
    
        rewind(fptr);

        long long idx = 0;
        while(true)
        {
            int flag = 0;
            int u,v;
            float r;

            flag += fread(&u, sizeof(int), 1, fptr); 
            flag += fread(&v, sizeof(int), 1, fptr); 
            flag += fread(&r, sizeof(float), 1, fptr); 
            if(flag != 3)break;

            if(u + 1 > prob.m)prob.m = u + 1;
            if(v + 1 > prob.n)prob.n = v + 1;

            R[idx].u = u;
            R[idx].v = v;
            R[idx].rate = r;
            idx ++;
            //if(idx > 0 && idx%100000000 == 0)printf("progress: %%%.3f\n",100.0*idx/prob.nnz);
        }
        prob.R = R;

        fclose(fptr);

        printf("m:%d, n:%d, nnz:%lld\n",prob.m, prob.n, prob.nnz);
        printf("time elapsed:%.8fs\n",(clock()-start)/double(CLOCKS_PER_SEC));
    }
    else
    {
        //data
        long long size_list[128];
        long long offset_list[128];
        pthread_t threads[128];
        pthread_arg pthread_arg_list[128];
    
        //get nnz & size_list
        FILE*fptrs[80];
        prob.nnz = 0;
        for(int i = 0;i < num_files;i++)
        {
            fptrs[i] = fopen(file_names[i].c_str(), "rb");
            fseek(fptrs[i], 0L, SEEK_END);
            size_list[i] = ftell(fptrs[i])/12;
            prob.nnz +=  size_list[i];
            fclose(fptrs[i]);
        }
    
        //get offset_list
        for(int i = 1;i < num_files;i++)
        {
            offset_list[i] = offset_list[i-1] + size_list[i-1];
        }

        //malloc
        mf_node *R;
        cudaMallocHost(&R,sizeof(mf_node)*prob.nnz); 
        prob.R = R;

        //launch
        for(int i = 0;i < num_files; i++)
        {
            pthread_arg_list[i].thread_id = i;
            pthread_arg_list[i].path = file_names[i];
            pthread_arg_list[i].R = prob.R;
            pthread_arg_list[i].offset = offset_list[i];
            pthread_arg_list[i].size = size_list[i];
            pthread_create(&(threads[i]), NULL, read_problem_thread, (void*)(&(pthread_arg_list[i])));
        }

        for(int i = 0;i < num_files;i++)
        {
            pthread_join(threads[i], NULL);
        }
        prob.m = -1;
        prob.n = -1;
        for(int i = 0;i < num_files;i++)
        {
            if(pthread_arg_list[i].max_m >= prob.m) prob.m = pthread_arg_list[i].max_m;
            if(pthread_arg_list[i].max_n >= prob.n) prob.n = pthread_arg_list[i].max_n;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = end.tv_sec - begin.tv_sec;
        elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;

        printf("m:%d, n:%d, nnz:%lld\n",prob.m, prob.n, prob.nnz);
        //printf("time elapsed:%.8fs\n",(clock()-start)/double(CLOCKS_PER_SEC));
        printf("time elapsed:%.8fs\n\n\n",elapsed);
    }

    return prob;
}

/*
mf_problem read_problem(string path)
{
    clock_t start;
    printf("-----read_problem called-----\n");
    //A simple function that reads the sparse matrix in COO manner.
    start = clock();

    mf_problem prob;
    prob.m = 1;
    prob.n = 1;
    prob.nnz = 0;
    prob.R = NULL;

    if(path.empty())
    {
        printf("file %s open failed\n", path.c_str());
        exit(0);
        return prob;
    }

    FILE*fptr = fopen(path.c_str(), "rb");
    if(fptr == NULL)
    {
        printf("file %s open failed\n", path.c_str());
        exit(0);
        return prob;
    }
    fseek(fptr, 0L, SEEK_END);
    prob.nnz = ftell(fptr)/12;

    mf_node *R;
    cudaMallocHost(&R,sizeof(mf_node)*prob.nnz); 
    gpuErr(cudaPeekAtLastError());

    rewind(fptr);

    long long idx = 0;
    while(true)
    {
        int flag = 0;
        int u,v;
        float r;

        flag += fread(&u, sizeof(int), 1, fptr); 
        flag += fread(&v, sizeof(int), 1, fptr); 
        flag += fread(&r, sizeof(float), 1, fptr); 

        if(flag != 3)break;

        if(u + 1 > prob.m)prob.m = u + 1;
        if(v + 1 > prob.n)prob.n = v + 1;

        R[idx].u = u;
        R[idx].v = v;
        R[idx].rate = r;
        idx ++;

        //if(idx > 0 && idx%100000000 == 0)printf("progress: %%%.3f\n",100.0*idx/prob.nnz);
    }
    prob.R = R;

    fclose(fptr);

    printf("read_problem:%s\n",path.c_str());
    printf("m:%d, n:%d, nnz:%lld\n",prob.m, prob.n, prob.nnz);
    printf("time elapsed:%.8fs\n",(clock()-start)/double(CLOCKS_PER_SEC));
    printf("-----read problem finished-----\n\n\n");
    
    return prob;
}
*/


void grid_problem(mf_problem* prob)
{
    clock_t start;
    
    printf("grid problem ...\n");
    fflush(stdout);

    //grid the problem into several grids
    long long u_seg, v_seg;
    if(prob->ux == 1)u_seg = prob->m;
    else u_seg = (long long)ceil((double)prob->m/prob->ux);
    if(prob->vy == 1)v_seg = prob->n;
    else v_seg = (long long)ceil((double)prob->n/prob->vy);

    prob->u_seg = u_seg;
    prob->v_seg = v_seg;
    

    auto get_grid_id = [=](int u, int v)
    {
        return ((u/u_seg)*prob->vy + v/v_seg);
    };

    //count the size of each grid
    prob->gridSize = new long long[prob->ux*prob->vy]();
    
    start = clock();
    long long *gridSize = prob->gridSize;
    for(long long i = 0;i < prob->nnz;i++)
    {
        int tmp_u = prob->R[i].u;
        int tmp_v = prob->R[i].v;
        gridSize[get_grid_id(tmp_u, tmp_v)] ++;
    }

    long long max_grid_size = 0;
    for(int i = 0;i < prob->ux*prob->vy; i++)
    {
        //printf("gridSize[%d]:%lld\n",i,prob->gridSize[i]);
        if(max_grid_size < prob->gridSize[i])max_grid_size = prob->gridSize[i];
    }
    prob->maxGridSize = max_grid_size;

    //generate the pointer to each grid.
    mf_node**R2D = new mf_node*[prob->ux*prob->vy + 1];
    mf_node* R = prob->R;
    R2D[0] = R;
    for(int grid = 0;grid < prob->ux*prob->vy; grid++)R2D[grid + 1] = R2D[grid] + gridSize[grid];
        
    prob->R2D = R2D;
    
    //swap
    printf("swapping ...\n");
    start = clock();
    mf_node**pivots = new mf_node*[prob->ux*prob->vy];
    for(int i = 0;i < prob->ux*prob->vy; i++)pivots[i] = R2D[i];

    for(int grid = 0; grid < prob->ux*prob->vy; grid++)
    {
        for(mf_node*pivot = pivots[grid]; pivot != R2D[grid + 1];)
        {
            int corre_grid = get_grid_id(pivot->u, pivot->v);
            if(corre_grid == grid)
            {  
                pivot ++;
                continue;
            }
            mf_node *next = pivots[corre_grid];
            swap(*pivot, *next);
            pivots[corre_grid] ++;
        }
    }

    printf("grid swap time:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n");
    fflush(stdout);
}



__global__ void init_rand_state(curandState*state)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init(clock() + tid, tid, 0,&state[tid]);
}

__global__ void random_init(
    curandState*state,
    int state_size,
    half*array,
    long long array_size,
    long long k, 
    float scale)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int state_id = tid%state_size;

    for(int i = 0;i < array_size;i += gridDim.x*blockDim.x)
    {
        int idx = i + tid;
        if(idx >= array_size)break;
        array[idx] = __float2half(curand_uniform(&state[state_id])*scale);
    }
}

void init_feature(short *feature_vec, int grid, long long seg, int k)
{

    float scale = (float)sqrt(1.0/k);

    half*gpu_vec;
    cudaMalloc(&gpu_vec, seg*k*sizeof(half));
    
    curandState* d_state;
    int state_size = (seg/256 + 1)*256;
    printf("state_size:%d\n", state_size);

    cudaMalloc(&d_state,sizeof(curandState)*state_size);

    init_rand_state<<<state_size/256, 256>>>(d_state);

    for(int i = 0;i < grid; i++)
    {
        printf("grid:%d\n",i);
        int blockSize = 256, blockNum = 1;
        blockNum = (seg*k + 255)/256;
        //printf("\tblockNum:%d\n", blockNum);
        //printf("\tarraysize:%lld\n", seg*k);

        random_init<<<blockNum, blockSize>>>(d_state, state_size, gpu_vec, seg*k, k, scale);
        cudaMemcpy(feature_vec + i*seg*k,gpu_vec,sizeof(half)*seg*k, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_state);
    cudaFree(gpu_vec);
}

mf_model* init_model(mf_problem*prob, int k, float ave)
{

    printf("init model ...\n");
    clock_t start = clock();

    mf_model *model = new mf_model;
    float scale_factor = sqrt(1.0/k);
    model->fun = 0;
    model->m = prob->m;
    model->n = prob->n;
   
    model->u_grid = prob->u_grid;
    model->v_grid = prob->v_grid;
   
    model->x_grid = prob->x_grid;
    model->y_grid = prob->y_grid;

    model->ux = prob->ux;
    model->vy = prob->vy;

    model->u_seg = prob->u_seg;
    model->v_seg = prob->v_seg;
    model->k = k;
    model->b = ave;

    //allocate memory
    cudaMallocHost(&model->floatp, sizeof(float)*model->ux*model->u_seg*k);
    cudaMallocHost(&model->floatq, sizeof(float)*model->vy*model->v_seg*k);

    cudaMallocHost(&model->halfp, sizeof(short)*model->ux*model->u_seg*k);
    cudaMallocHost(&model->halfq, sizeof(short)*model->vy*model->v_seg*k);

    gpuErr(cudaPeekAtLastError());

    //random init
    init_feature(model->halfp, model->ux, model->u_seg, k);
    init_feature(model->halfq, model->vy, model->v_seg, k);

    printf("time elapsed:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n\n");
    
    return model;
}

void scale_model(mf_model *model, float scale)
{
    printf("scale model ...\n");
    clock_t start = clock();

    float factor_scale = sqrt(scale);
    for(long long i = 0; i < ((long long)model->m)*model->k; i++)model->floatp[i] = model->floatp[i]*factor_scale;


    for(long long i = 0; i < model->n*model->k; i++)model->floatq[i] = model->floatq[i]*factor_scale;
    
    printf("time elapsed:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n\n");
}


void shuffle_model(mf_model *model, int* inv_p_map, int* inv_q_map)
{
    auto inv_shuffle1 = [] (float *vec,
                            int *map,
                            int size,
                            int k)
    {
        for(int pivot = 0; pivot < size;)
        {
            if(pivot == map[pivot])
            {
                ++pivot;
                continue;
            }

            int next = map[pivot];

            for(SGDIndex d = 0; d < k; d++)swap(*(vec + (long long)pivot*k+d), *(vec+(long long)next*k+d));

            map[pivot] = map[next];
            map[next] = next;
        }
    };

    inv_shuffle1(model->floatp, inv_p_map, model->m, model->k);
    inv_shuffle1(model->floatq, inv_q_map, model->n, model->k);
}

//the core computation function
mf_model*sgd_train(mf_problem*tr, mf_problem*te, Parameter para)
{

    clock_t start;

    //collect the factor. scaling is used to make sure every rating is around 1.
    SGDRate ave;
    SGDRate std_dev;
    SGDRate scale = 1.0;

    collect_data(tr, ave, std_dev);
    scale = max((SGDRate)1e-4, std_dev);

    fflush(stdout);

    //shuffle the u & v randomly to: 1) increase randomness. 2) block balance.
    printf("shuffle problem ...\n");
    start = clock();
    int* p_map = gen_random_map(tr->m);
    int* q_map = gen_random_map(tr->n);
    int* inv_p_map = gen_inv_map(p_map, tr->m);
    int* inv_q_map = gen_inv_map(q_map, tr->n);

    shuffle_problem(tr, p_map, q_map);

    printf("time elapsed:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n\n");

    //problem grid.
    grid_problem(tr); 

    //scale problem
    printf("scale problem ...\n");
    start = clock();
    scale_problem(tr, 1.0/scale, tr->u_seg, tr->v_seg);
    para.lambda_p = para.lambda_p/scale;
    para.lambda_q = para.lambda_q/scale;
    printf("time elapsed:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n\n");

    //init model
    mf_model*model = init_model(tr, para.k, ave/std_dev);

    //train
    sgd_update_k128(para, model, tr, scale);

    //scale model
    scale_model(model, scale);

    //shuffle model
    shuffle_model(model, inv_p_map, inv_q_map);
    return model;
}
