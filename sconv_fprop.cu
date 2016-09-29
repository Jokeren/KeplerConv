#include <vector>
#include <string>
#include <map>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

std::map<std::string, CUfunction> nervana_kernels;
std::vector<CUmodule> nervana_modules;

int len_d2b(int n)
{
  int i, j = 0;
  i = n;
  while (i) {
    i /= 2;
    j++;
  }
  return j;
}

void magic32(unsigned int nmax, unsigned int d, unsigned int& m, unsigned int& p)
{
  long nc = ((nmax + 1) / d) * d - 1;
  long nbits = len_d2b(nmax);
  for(long p = 0; p < 2 * nbits + 1; p++) {   
    if(pow(2, p) > nc * (d - 1 - (long)(pow(2, p) - 1) % d)) {
      m = (pow(2, p) + d - 1 -(long)(pow(2, p) - 1) % d) / d;
      return;
    }   
  }   
  return;
}

void magic64(unsigned int d, unsigned int& magic, unsigned int& shift) {
  // 3 is a special case that only ends up in the high bits
  // if the nmax is 0xffffffff
  // we can't use 0xffffffff for all cases as some return a 33 bit
  // magic number
  unsigned long nmax;
  if(d == 3)
    nmax = 0xffffffff;
  else
    nmax = 0x7fffffff;
  magic32(nmax, d, magic, shift);
  if(magic != 1)
    shift -= 32;
}

bool load_kernels(const char* const base_path_cstr) {
    //better would be a vector<string>, but there is a bug in nvcc that prevents this
    // (bug report filed)
    std::string names[1] = {
        "sconv_fprop_K64_N64",
    };

    std::string base_path(base_path_cstr);

    for (int i = 0; i < 1; ++i) {
      std::string kernel = names[i];
        if (nervana_kernels.count(kernel) > 0)
            continue;

        CUmodule module;

        std::string path = base_path + kernel + std::string(".cubin");
        CUresult res = cuModuleLoad(&module, path.c_str());

        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to load: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_modules.push_back(module);

        CUfunction function;
        res = cuModuleGetFunction(&function, module, kernel.c_str());
        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to extract: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_kernels.insert(std::make_pair(kernel, function));
    }

    return true;
}

bool fprop(const float *I, const float *F, float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
  float alpha = 1.0f;
  unsigned int WN, HWN, DHWN, KRST, RST, RS;
  unsigned int PQ, QN, PQN, MPQN;
  unsigned int magic_RS, magic_S;
  unsigned int shift_RS, shift_S;
  unsigned int magic_Q, shift_Q, magic_PQ, shift_PQ;

  WN = W * N;
  HWN = H * WN;
  DHWN = D * HWN;
  RS = R * S;
  RST = T * RS;
  KRST = K * RST;

  QN = Q * N;
  PQN = P * QN;
  MPQN = M * PQN;
  PQ = P * Q;

  magic64(Q, magic_Q, shift_Q);
  magic64(PQ, magic_PQ, shift_PQ);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);

  float *test_param;
  cudaError_t result;
  result = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  if (result != cudaSuccess) {
    std::cerr << "cuda malloc error!" << std::endl;
    exit(1);
  }
  void *args[37] = {&test_param, &O, &I, &F, &alpha,
    &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
    &C, &KRST, &RST, &RS, &magic_RS, &shift_RS, &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
    &Q, &PQ, &QN, &PQN, &MPQN, &magic_Q, &shift_Q, &magic_PQ, &shift_PQ};
  int gridMPQ = M * P * Q;
  int gridX = gridMPQ;
  int gridY = K / 64 + (K % 64 != 0);
  int gridZ = N / 64 + (N % 64 != 0);

  std::string name = "sconv_fprop_K64_N64";
  CUresult res = cuLaunchKernel(nervana_kernels[name], gridX, gridY, gridZ, 64, 1, 1, 0, 0, args, NULL);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Error launching kernel " << name << " " << res << std::endl;
    return false;
  }
  cuCtxSynchronize();
  //float* h_test = (float *)malloc(sizeof(float) * 64);
  //for (int i = 0; i < 64; ++i) {
  //  std::cout << h_test[i] << " ";
  //}
  //std::cout << std::endl;
  //result = cudaMemcpy(h_test, test_param, sizeof(float) * 64, cudaMemcpyDeviceToHost);
  //if (result != cudaSuccess) {
  //  std::cout << result << std::endl;
  //  std::cerr << "memcpy error!" << std::endl;
  //}

  //for (int i = 0; i < 64; ++i) {
  //  std::cout << h_test[i] << " ";
  //}

  //free(h_test);

  //std::cout << std::endl;

  return true;
}

int main() {
  cudaFree(0);
  float *d_I, *d_F, *d_O;
  unsigned int N = 64, C = 1, K = 64, D = 1, H = 5, W = 5, T = 1, R = 5, S = 5;
  unsigned int str_d = 1, str_h = 1, str_w = 1;
  unsigned int pad_d = 0, pad_h = 0, pad_w = 0;
  unsigned int M, P, Q;
  M = (D - T + 2 * pad_d) / str_d + 1;
  P = (H - R + 2 * pad_h) / str_h + 1;
  Q = (W - S + 2 * pad_w) / str_w + 1;
  std::cout << " M " << M << std::endl;
  std::cout << " P " << P << std::endl;
  std::cout << " Q " << Q << std::endl;

  float *h_I = (float *)malloc(N * C * D * H * W * sizeof(float));
  for (int i = 0; i < N * C * D * H * W; ++i) {
    h_I[i] = 1;
  }
  float *h_F = (float *)malloc(K * R * S * T * sizeof(float));
  for (int i = 0; i < K * R * S * T; ++i) {
    h_F[i] = 1;
  }

  cudaMalloc((void**)&d_I, sizeof(float) * N * C * D * H * W);
  cudaMalloc((void**)&d_F, sizeof(float) * K * R * S * T);
  cudaMalloc((void**)&d_O, sizeof(float) * K * M * P * Q * N);
  float* h_O = (float *)malloc(sizeof(float) * K * M * P * Q * N);
  std::cout << "before" << std::endl;

  cudaError_t result = cudaMemcpy(h_O, d_O, sizeof(float) * K * M * P * Q * N, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cout << result << std::endl;
    std::cerr << "memcpy error!" << std::endl;
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << h_O[i] << " ";
  }
  std::cout << std::endl;

  cudaMemcpy(d_I, h_I, sizeof(float) * N * C * D * H * W,
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, sizeof(float) * K * R * S * T,
    cudaMemcpyHostToDevice);

  if (!load_kernels("./")) {
    std::cerr << "Couldn't load all kernels" << std::endl;
    exit(1);
  }


  if (!fprop(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
    std::cerr << "Launch error" << std::endl;
  }

  std::cout << "result" << std::endl;

  result = cudaMemcpy(h_O, d_O, sizeof(float) * K * M * P * Q * N, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cout << result << std::endl;
    std::cerr << "memcpy error!" << std::endl;
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << h_O[i] << " ";
  }
  
  std::cout << std::endl;

  free(h_O);
  free(h_I);
  free(h_F);
  cudaFree(d_I);
  cudaFree(d_F);
  cudaFree(d_O);

  std::cout << "finish" << std::endl;

  return 0;
}
