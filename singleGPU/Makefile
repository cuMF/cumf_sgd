


CU_FLAG=  -gencode arch=compute_35,code=compute_35 \
	      -gencode arch=compute_37,code=compute_37 \
          -gencode arch=compute_50,code=compute_50

cumf_sgd:main.o sgd.o sgd_kernel.o
	nvcc main.o sgd.o sgd_kernel.o -o cumf_sgd -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt

main.o:main.cpp sgd.h 
	nvcc -c main.cpp -std=c++11 -DUSEOMP -Xcompiler -fopenmp -O3 -I /usr/local/cuda/include -Xcompiler -g

sgd.o:sgd.cu sgd.h
	nvcc -c sgd.cu -o sgd.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -march=native -DUSEOMP $(CU_FLAG) -I /usr/local/cuda/include -Xcompiler -g -lpthread

sgd_kernel.o:sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -c sgd_kernel.cu -o sgd_kernel.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -march=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g

clean:
	rm -f cumf_sgd *.o
