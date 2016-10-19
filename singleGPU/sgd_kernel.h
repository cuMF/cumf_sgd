#ifndef _SGD_KERNEL_H_
#define _SGD_KERNEL_H_

#include "sgd.h"

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void sgd_update_k128(Parameter para, mf_model *model, mf_problem *prob, float scale);


#endif