#include "sconv.h"

bool update(const float *I, float *F, const float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
}

int main() {
  cudaFree(0);
  float *d_I, *d_F, *d_O;
  unsigned int N = 64, C = 64, K = 1, D = 1, H = 5, W = 5, T = 1, R = 5, S = 5;
  unsigned int str_d = 1, str_h = 1, str_w = 1;
  unsigned int pad_d = 0, pad_h = 0, pad_w = 0;
  unsigned int M, P, Q;
  M = (D - T + 2 * pad_d) / str_d + 1;
  P = (H - R + 2 * pad_h) / str_h + 1;
  Q = (W - S + 2 * pad_w) / str_w + 1;
  float *h_O = (float *)malloc(N * M * P * Q * K * sizeof(float));
  for (int i = 0; i < N * M * P * Q * K; ++i) {
    h_O[i] = 1;
  }
  float *h_I = (float *)malloc(K * D * H * W * N * sizeof(float));
  for (int i = 0; i < K * D * H * W * N; ++i) {
    h_I[i] = 1;
  }
  float* h_F = (float *)malloc(sizeof(float) * K * R * S * T);
  cudaMalloc((void**)&d_I, sizeof(float) * N * C * D * H * W);
  cudaMalloc((void**)&d_F, sizeof(float) * K * R * S * T);
  cudaMalloc((void**)&d_O, sizeof(float) * K * M * P * Q * N);
  cudaMemcpy(d_I, h_I, sizeof(float) * N * C * D * H * W,
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_O, h_O, sizeof(float) * N * M * P * Q * K,
    cudaMemcpyHostToDevice);

  if (!update(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
    std::cerr << "Launch error" << std::endl;
  }

  std::cout << "result" << std::endl;

  cudaError_t result = cudaMemcpy(h_F, d_F, sizeof(float) * C * K * R * S * T, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cout << result << std::endl;
    std::cerr << "memcpy error!" << std::endl;
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << h_F[i] << " ";
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
