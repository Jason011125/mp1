#include <chrono>
#include "../include/utils.h"
using namespace std;

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  for (int j = 0; j < Ref::N; j++) {		\
    for (int i = 0; i < Ref::M; i++) {    		\
      refC[i * Ref::N + j] = 0;		\
	}		\
  }		\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								\
  fillRandom(refC, Ref::M * Ref::N);				
  
#define TIME(name) \
  for (int i = 0; i < 2; i++)						\
    {									\
	  for (int j = 0; j < N; j++) {		\
	    for (int k = 0; k < M; k++) {    		\
	      C[k * N + j] = 0;		\
	    }		\
	  }		\
      name(A, B, C, M, N, K);						\
    }									\
  double time_ ## name; \
  for (int i = 0; i < 3; i++)						\
    {									\
	  for (int j = 0; j < N; j++) {		\
	    for (int k = 0; k < M; k++) {    		\
	      C[k * N + j] = 0;		\
	    }		\
	  }		\
	  auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K);						\
	  auto end_time_ ## name = std::chrono::high_resolution_clock::now();	\
  	  std::chrono::duration<double, std::milli> duration_ ## name = end_time_ ## name - start_time_ ## name; \
	  time_ ## name += duration_ ## name.count(); \
    }									\
  std::cout << "Time taken for GEMM (CPU," << #name <<"): " << time_ ## name << "ms" << std::endl; 


// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
	C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler flags. So, you can use o3's code for that part
void gemm_cpu_o1(float* A, float* B, float *C, int M, int N, int K) {
	for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k]; 
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

void gemm_cpu_o2(float* A, float* B, float *C, int M, int N, int K) {
	int TILE_SIZE = 32;
	for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k += TILE_SIZE) {  
            for (int j = 0; j < N; j += TILE_SIZE) { 

                int minK = std::min(k + TILE_SIZE, K);
                int minN = std::min(j + TILE_SIZE, N);

                for (int k_t = k; k_t < minK; k_t++) {  
                    float a_ik = A[i * K + k_t];  
                    for (int j_t = j; j_t < minN; j_t++) {  
                        C[i * N + j_t] += a_ik * B[k_t * N + j_t]; 
                    }
                }
            }
        }
    }

}

void gemm_cpu_o3(float* A, float* B, float *C, int M, int N, int K) {
	#pragma omp parallel for
	for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k]; 
			#pragma omp simd
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}


int main(int argc, char* argv[]) {
	if (argc < 3) {
	  std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
	  return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	// Check if the kernel results are correct
	// note that even if the correctness check fails all optimized kernels will run.
	// We are not exiting the program at failure at this point.
	// It is a good idea to add more correctness checks to your code.
	// We may (at discretion) verify that your code is correct.
	float* refC = new float[Ref::M * Ref::N]();
	auto ref = Ref();
	CHECK(gemm_cpu_o0)
	CHECK(gemm_cpu_o1)
	CHECK(gemm_cpu_o2)
	CHECK(gemm_cpu_o3)
	delete[] refC;
	
	TIME(gemm_cpu_o0)
	TIME(gemm_cpu_o1)
	TIME(gemm_cpu_o2)
	TIME(gemm_cpu_o3)

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
