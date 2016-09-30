#include "sconv.h"

bool bprop(float *I, const float *F, const float *I,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
  float alpha = 1.0f;
  unsigned int DHW, WN, HW, HWN, DHWN, CRST, RST, RS;
  unsigned int MPQ, PQ, QN, PQN, MPQN;
  unsigned int magic_HW, magic_W;
  unsigned int shift_HW, shift_W;
  unsigned int magic_RST, magic_RS, magic_S;
  unsigned int shift_RST, shift_RS, shift_S;
  unsigned int magic_Q, shift_Q, magic_PQ, shift_PQ;
  unsigned int magic_str_w, magic_str_h, magic_str_d;
  unsigned int shift_str_w, shift_str_h, shift_str_d;

  WN = W * N;
  HW = H * W;
  HWN = H * WN;
  DHW = D * HW;
  DHWN = D * HWN;
  RS = R * S;
  RST = T * RS;
  CRST = C * RS;

  QN = Q * N;
  PQN = P * QN;
  MPQN = M * PQN;
  PQ = P * Q;
  MPQ = M * P * Q;

  magic32(MPQ, PQ, magic_PQ, shift_PQ);
  magic32(PQ, Q, magic_Q, shift_Q);
  magic32(CRST, RST, magic_RST, shift_RST);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);
  magic32(W + S - pad_w - 2, str_w, magic_str_w, shift_str_w);
  magic32(H + R - pad_h - 2, str_h, magic_str_h, shift_str_h);
  magic32(D + T - pad_d - 2, str_d, magic_str_d, shift_str_d);
  magic32(DHW, HW, magic_HW, shift_HW);
  magic32(HW, W, magic_W, shift_W);

  float *test_param;
  cudaError_t result;
  result = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  if (result != cudaSuccess) {
    std::cerr << "cuda malloc error!" << std::endl;
    exit(1);
  }
  void *args[40] = {&test_param, &I, &O, &F, &alpha,
    &N, &C, &M, &P, &Q, &QN, &PQN, &MPQN,
    &K, &CRST, &RST, &RS, &magic_RS, &shift_RS, &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
    &W, &HW, &WN, &HWN, &DHWN, &magic_W, &shift_HW,
    &R, &T, &magic_str_w, &magic_str_h, &magic_str_d};
  int gridDWH = D * W * H;
  int gridX = gridDWH;
  int gridY = C / 64 + (C % 64 != 0);
  int gridZ = N / 64 + (N % 64 != 0);

  std::string name = "sconv_fprop_K64_N64";
  CUresult res = cuLaunchKernel(nervana_kernels[name], gridX, gridY, gridZ, 64, 1, 1, R * S * T * 4 * 2, 0, args, NULL);
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
  float *h_F = (float *)malloc(K * R * S * T * sizeof(float));
  for (int i = 0; i < K * R * S * T; ++i) {
    h_F[i] = 1;
  }
  float* h_I = (float *)malloc(sizeof(float) * K * M * P * Q * N);
  cudaMalloc((void**)&d_I, sizeof(float) * N * C * D * H * W);
  cudaMalloc((void**)&d_F, sizeof(float) * K * R * S * T);
  cudaMalloc((void**)&d_O, sizeof(float) * K * M * P * Q * N);
  cudaMemcpy(d_O, h_O, sizeof(float) * N * M * P * Q * K,
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, sizeof(float) * K * R * S * T,
    cudaMemcpyHostToDevice);

  if (!bprop(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
    std::cerr << "Launch error" << std::endl;
  }

  std::cout << "result" << std::endl;

  cudaError_t result = cudaMemcpy(h_I, d_I, sizeof(float) * C * D * H * W * N, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cout << result << std::endl;
    std::cerr << "memcpy error!" << std::endl;
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << h_I[i] << " ";
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
