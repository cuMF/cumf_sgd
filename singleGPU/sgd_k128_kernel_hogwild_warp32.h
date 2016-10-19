

__global__ void sgd_k128_kernel_hogwild_warp32_lrate(
                            const mf_node *R,
                            long long nnz,
                            half *p,
                            half *q,
                            curandState *state,
                            float *dynamic_rate,
                            long long u_seg,
                            long long v_seg,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_per_block, 
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_p,
                            float lambda_q,
                            double *gpu_iter_err,
                            int u_grid,
                            int v_grid,
                            int u_id,
                            int v_id
                            )
{


    //persistant thread
    for(int ite = current_iter; ite < current_iter + num_iters; ite ++)
    {
        float tmp_lrate = __ldg(&dynamic_rate[ite]);
        
        for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
        {

            long long start_id;
            if(threadIdx.x == 0)
            {
                long long origin = (long long)(curand_uniform(&state[blockIdx.x])*nnz);
                start_id = origin%nnz;
            }
            start_id = __shfl(start_id, 0);
            
            for(int i = 0;i < update_vector_size;i++)
            {
                int offset = (start_id + i)%nnz;
                
                float e = __ldg(&R[offset].rate);
                int u = __ldg(&R[offset].u);
                int v = __ldg(&R[offset].v);

                //read the p & q into register file.
                int base_p = u*k;
                int base_q = v*k;

                float tmp_p1 = __half2float(p[base_p + threadIdx.x]);
                float tmp_q1 = __half2float(q[base_q + threadIdx.x]);
            
                float tmp_p2 = __half2float(p[base_p + threadIdx.x + 32]);
                float tmp_q2 = __half2float(q[base_q + threadIdx.x + 32]);
            
                float tmp_p3 = __half2float(p[base_p + threadIdx.x + 64]);
                float tmp_q3 = __half2float(q[base_q + threadIdx.x + 64]);
            
                float tmp_p4 = __half2float(p[base_p + threadIdx.x + 96]);
                float tmp_q4 = __half2float(q[base_q + threadIdx.x + 96]);

                float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

                //get dot product.
                tmp_product += __shfl_down(tmp_product, 16);
                tmp_product += __shfl_down(tmp_product, 8);
                tmp_product += __shfl_down(tmp_product, 4);
                tmp_product += __shfl_down(tmp_product, 2);
                tmp_product += __shfl_down(tmp_product, 1);

                tmp_product = __shfl(tmp_product,0);

                float ruv = e - tmp_product;

                #ifdef PRINTITE
                    block_err += ruv*ruv;
                #endif

                //update
                //only works for k=blockDim.x=128
                p[base_p + threadIdx.x +  0] = __float2half(tmp_p1 + tmp_lrate*(ruv*tmp_q1 - lambda_p*tmp_p1));
                q[base_q + threadIdx.x +  0] = __float2half(tmp_q1 + tmp_lrate*(ruv*tmp_p1 - lambda_q*tmp_q1));

                p[base_p + threadIdx.x + 32] = __float2half(tmp_p2 + tmp_lrate*(ruv*tmp_q2 - lambda_p*tmp_p2));
                q[base_q + threadIdx.x + 32] = __float2half(tmp_q2 + tmp_lrate*(ruv*tmp_p2 - lambda_q*tmp_q2));

                p[base_p + threadIdx.x + 64] = __float2half(tmp_p3 + tmp_lrate*(ruv*tmp_q3 - lambda_p*tmp_p3));
                q[base_q + threadIdx.x + 64] = __float2half(tmp_q3 + tmp_lrate*(ruv*tmp_p3 - lambda_q*tmp_q3));

                p[base_p + threadIdx.x + 96] = __float2half(tmp_p4 + tmp_lrate*(ruv*tmp_q4 - lambda_p*tmp_p4));
                q[base_q + threadIdx.x + 96] = __float2half(tmp_q4 + tmp_lrate*(ruv*tmp_p4 - lambda_q*tmp_q4));
            }    
        }
    }
    
}



