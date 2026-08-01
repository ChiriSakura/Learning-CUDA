#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

//--------------------------------------------------------------------
//                            RMSNorm Kernel                          
//--------------------------------------------------------------------
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input, const T* __restrict__ weight, T* __restrict__ output,
                              size_t rows, size_t hidden_dim, float eps) {
  // 每个Block处理一行
  const size_t row = blockIdx.x;
  const T* row_in = input + row * hidden_dim;
  T* row_out = output + row * hidden_dim;

  // 平方和统一在 float 精度下累加，避免 half 的累积误差与转换歧义
  float local_sum = 0.0f;
  for (size_t j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float val = static_cast<float>(__ldg(&row_in[j]));
    local_sum += val * val;
  }

  // Warp 内归约求和
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  } 

  // Warp 间归约求和
  __shared__ float shared_sum[32];
  const int lane = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;

  if (lane == 0) {
    shared_sum[warp_id] = local_sum;
  }
  __syncthreads();

  // 最终归约求和，并计算 rms 值
  __shared__ float rms;
  if (warp_id == 0) {
    const int num_warps = blockDim.x / warpSize;
    local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
      float mean_sq = local_sum / static_cast<float>(hidden_dim);
      rms = rsqrtf(mean_sq + eps);
    }
  }
  __syncthreads();

  // 使用 rms 和权重计算输出
  for (size_t j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float val = static_cast<float>(__ldg(&row_in[j]));
    float w = static_cast<float>(__ldg(&weight[j]));
    row_out[j] = static_cast<T>(val * rms * w);
  }
}


/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
namespace {
// 复用的设备缓冲区：按需增长，避免每次调用重复 cudaMalloc/cudaFree 的固定开销
void* g_rmsBuf = nullptr;
size_t g_rmsBufBytes = 0;

void* getRmsDeviceBuffer(size_t bytes) {
  if (bytes > g_rmsBufBytes) {
    if (g_rmsBuf) cudaFree(g_rmsBuf);
    cudaMalloc(&g_rmsBuf, bytes);
    g_rmsBufBytes = bytes;
  }
  return g_rmsBuf;
}
}  // namespace

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  // 边界处理
  if (rows == 0 || hidden_dim == 0 || h_input.empty() || h_weight.empty()) {
    return;
  }

  // 确保输出大小正确
  h_output.resize(rows * hidden_dim);

  // 单块设备内存分三段（input / weight / output），按 256B 对齐，整体复用
  const size_t in_bytes = rows * hidden_dim * sizeof(T);
  const size_t w_bytes = hidden_dim * sizeof(T);
  const size_t out_bytes = rows * hidden_dim * sizeof(T);
  const size_t align = 256;
  auto roundUp = [](size_t x, size_t a) { return (x + a - 1) / a * a; };
  const size_t off_weight = roundUp(in_bytes, align);
  const size_t off_output = roundUp(off_weight + w_bytes, align);
  const size_t total_bytes = off_output + out_bytes;

  char* base = static_cast<char*>(getRmsDeviceBuffer(total_bytes));
  T* d_input = reinterpret_cast<T*>(base);
  T* d_weight = reinterpret_cast<T*>(base + off_weight);
  T* d_output = reinterpret_cast<T*>(base + off_output);

  // 将数据从主机复制到设备
  cudaMemcpy(d_input, h_input.data(), in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), w_bytes, cudaMemcpyHostToDevice);

  // 调用 CUDA 核函数, 配置每个 Block 处理一行
  constexpr size_t BLOCK_SIZE = 256;
  size_t grid = rows;

  rmsNormKernel<T><<<grid, BLOCK_SIZE>>>(d_input, d_weight, d_output, rows, hidden_dim, eps);

  // 将结果从设备复制回主机
  cudaMemcpy(h_output.data(), d_output, out_bytes, cudaMemcpyDeviceToHost);
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
