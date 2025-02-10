// Creator.h
#include "util.h"
#include "ori_kernel/cp_kernel.h"
#include "ori_kernel/cutcp_kernel.h"
#include "ori_kernel/fft_kernel.h"
#include "ori_kernel/lbm_kernel.h"
#include "ori_kernel/mrif_kernel.h"
#include "ori_kernel/mriq_kernel.h"
#include "ori_kernel/sgemm_kernel.h"
#include "ori_kernel/stencil_kernel.h"
#include "tzgemm_kernel.h"

#include "ori_kernel/lava_kernel.h"
#include "ori_kernel/hot3d_kernel.h"
#include "ori_kernel/nn_kernel.h"
#include "ori_kernel/path_kernel.h"

#include "GPTBKernel.h"
#include "MixKernel.h"

#include <unordered_map>

using namespace std;

constexpr uint64_t FNV_prime = 16777619u;
constexpr uint64_t FNV_offset_basis = 2166136261u;

constexpr inline int myHash(const char* text);

GPTBKernel* createKernel(const std::string &name);

MixKernel* createMixKernel(const std::string &name);