'''
Author: diagonal
Date: 2023-11-15 22:45:42
LastEditors: diagonal
LastEditTime: 2023-12-05 23:28:24
FilePath: /tacker/mix_kernels/code/gen_mix.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
import re

from data import get_kernel_info, fuse_kernel_info
from util import extract_kernel_signature, process_parameter_list

def generate_mixed_kernel(kernel_info: list, blks_per_sm: int, ratio: list = None):
    """
    kernel_info: list of kernel signature
    ratio: list of ratio of each kernel, such as [1, 1, 2]
    """
    if ratio is None:
        ratio = [1] * len(kernel_info)

    if len(kernel_info) != len(ratio):        
        print("The length of kernel_info and ratio should be equal.")
        exit(1)
    mix_kernel_name = "mixed_"
    mix_kernel_params = ''
    mixed_kernel_body_code = ''

    thread_idx_threshold = 0
    for i, kernel_signature in enumerate(kernel_info):
        func_name, func_full_params = extract_kernel_signature(kernel_signature)
        params_list = process_parameter_list(func_name, i, func_full_params)
        # print(f"[{i}]func_name: ", func_name, end='\n\n')
        # print(f"[{i}]params_list: ", params_list, end='\n\n')
        mix_kernel_name += func_name + '_'
        current_kernel_params = ''
        for param in params_list[:-1]:
            mix_kernel_params += param[0] + ' ' + param[1] + ', '
            if param[1].endswith('_ptb_iter_block_step'):
                current_kernel_params += param[1] + f' * {ratio[i]}, '
            elif param[1].endswith('_ptb_start_block_pos'):
                current_kernel_params += param[1] + f' + $START_ARGS * {func_name}{i}_ptb_iter_block_step, '
            else:
                current_kernel_params += param[1] + ', '
        # print(f"[{i}]mix_kernel_params: ", mix_kernel_params, end='\n\n')
        # print(f"[{i}]current_kernel_params: ", current_kernel_params, end='\n\n')
        # print(f"[{i}]mix_kernel_params: ", mix_kernel_params, end='\n\n')
        for ii in range(ratio[i]):
            thread_idx_threshold += get_kernel_info(func_name)["blocksize"]
            if i + ii == 0:
                mixed_kernel_body_code += "    "
            mixed_kernel_body_code += f"if (threadIdx.x < {thread_idx_threshold}) {{\n"
            mixed_kernel_body_code += f"        {kernel_signature.split('(')[0].split(' ')[-1]}{ii}(\n"
            mixed_kernel_body_code += f"            {current_kernel_params}{thread_idx_threshold - get_kernel_info(func_name)['blocksize']}\n"
            mixed_kernel_body_code += f"        );\n"
            mixed_kernel_body_code += f"    }}\n"
            if i != len(kernel_info) - 1 or ii != ratio[i] - 1:
                mixed_kernel_body_code += f"    else "
            # print("body code: ", mixed_kernel_body_code)
            mixed_kernel_body_code = mixed_kernel_body_code.replace('$START_ARGS', str(ii), 1)

    mixed_kernel_body_code += """
}
"""
    mix_kernel_name += "kernel"

    for i in range(len(ratio)):
        mix_kernel_name += f"_{ratio[i]}"
    # print(f"mix_kernel_name: ", mix_kernel_name, end='\n\n')

    mix_kernel_signature = mix_kernel_name + '(' + mix_kernel_params[:-2] + ')'
    # print(f"mix_kernel_signature: ", mix_kernel_signature, end='\n\n')

    return "__global__ void " + mix_kernel_signature + "{\n" + mixed_kernel_body_code


# def generate_mixed_kernel(kernel_info):
#     mix_kernel_name = "mixed_"
#     mix_kernel_params = ''
#     mixed_kernel_body_code = ''
#     # mixed_kernel_code = f"""__global__ void mixed_$NAME_kernel({", ".join(params_names_list)}) """ + '{\n'
#     thread_idx_threshold = 0
#     for i, kernel_signature in enumerate(kernel_info):
#         func_name, func_full_params = extract_kernel_signature(kernel_signature)
#         params_list = process_parameter_list(func_name, i, func_full_params)
#         # print(f"[{i}]func_name: ", func_name, end='\n\n')
#         # print(f"[{i}]params_list: ", params_list, end='\n\n')
#         mix_kernel_name += func_name + '_'
#         current_kernel_params = ''
#         for param in params_list[:-1]:
#             mix_kernel_params += param[0] + ' ' + param[1] + ', '
#             current_kernel_params += param[1] + ', '
#         # print(f"[{i}]mix_kernel_params: ", mix_kernel_params, end='\n\n')

#         thread_idx_threshold += get_kernel_info(func_name)["blocksize"]
#         if i == 0:
#             mixed_kernel_body_code += "    "
#         mixed_kernel_body_code += f"if (threadIdx.x < {thread_idx_threshold}) {{\n"
#         mixed_kernel_body_code += f"        {kernel_signature.split('(')[0].split(' ')[-1]}(\n"
#         mixed_kernel_body_code += f"            {current_kernel_params}{thread_idx_threshold - get_kernel_info(func_name)['blocksize']}\n"
#         mixed_kernel_body_code += f"        );\n"
#         mixed_kernel_body_code += f"    }}\n"
#         if i != len(kernel_info) - 1:
#             mixed_kernel_body_code += f"    else "

#     mixed_kernel_body_code += """
# }
# """
#     mix_kernel_name += "kernel"
#     # print(f"mix_kernel_name: ", mix_kernel_name, end='\n\n')

#     mix_kernel_signature = mix_kernel_name + '(' + mix_kernel_params[:-2] + ')'
#     # print(f"mix_kernel_signature: ", mix_kernel_signature, end='\n\n')

#     return "__global__ void " + mix_kernel_signature + "{\n" + mixed_kernel_body_code

# 示例使用：三个 kernel，每个 kernel 有不同的 block dimension
kernel_signatures = {
    "cp": "__global__ void general_ptb_cp(int numatoms, float gridspacing, float* energygrid, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
    "cutcp": "__global__ void general_ptb_cutcp(int binDim_x, int binDim_y, float4* binZeroAddr, float h, float cutoff2, float inv_cutoff2, float* regionZeroAddr, int zRegionIndex_t, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
    "fft": "__global__ void general_ptb_fft(float2* data, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
    "lbm": "__global__ void general_ptb_lbm(float* srcGrid, float* dstGrid, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
    "mrif": "__global__ void general_ptb_mrif(int numK, int kGlobalIndex, float* x, float* y, float* z, float* outR, float* outI, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
    "mriq": "__global__ void general_ptb_mriq(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr , float* Qi, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
    "sgemm": "__global__ void general_ptb_sgemm(float* A, float* B, float* C, int NORMAL_M, int NORMAL_N, int NORMAL_K, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
    # "tpacf": "__global__ void general_ptb_tpacf(hist_t* histograms, float* all_x_data, float* all_y_data, float* all_z_data, int NUM_SETS, int NUM_ELEMENTS, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)"
    "stencil": "__global__ void general_ptb_stencil(float c0, float c1, float* A0, float* Anext, int nx, int ny, int nz, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)"
}

# 从kernel_signatures生成所有的[kernel1, kernel2]可能的pair
# kernel_pairs = []
# for i, kernel1 in enumerate(kernel_signatures):
#     for j, kernel2 in enumerate(kernel_signatures):
#         if i < j:
#             kernel_pairs.append([kernel1, kernel2])

# print('len(kernel_pairs): ', len(kernel_pairs))
# print('kernel_pairs: ', kernel_pairs)

# for kernel1, kernel2 in kernel_pairs:
#     # 判断文件是否存在
#     file_path = f"/home/jxdeng/workspace/tacker/ptb_kernels/mix_kernel/{kernel1}-{kernel2}.cu"
#     f = open(file_path, 'w')

#     print("kernel pair: ", kernel1, kernel2)
 
#     generated_list = [] # 去重

#     candidates = fuse_kernel_info(kernel1, kernel2)
#     for candidate in candidates:
#         blks_per_sm, ratio_1, ratio_2 = candidate[0], candidate[1], candidate[2]
#         print(f"// {kernel1}-{kernel2}-{ratio_1}-{ratio_2}", file=f)
#         mixed_kernel_code = generate_mixed_kernel([kernel_signatures[kernel1], kernel_signatures[kernel2]], blks_per_sm, [ratio_1, ratio_2])
#         print(mixed_kernel_code, file=f)
#         generated_list.append((ratio_1, ratio_2))
    
#     print("-", kernel1, "as main kernel:")
#     for candidate in candidates:
#         reg_per_thread = max(get_kernel_info(kernel1)["register"], get_kernel_info(kernel2)["register"])
#         print(f"-- {kernel1}_num:", candidate[1], f"{kernel2}_num:", candidate[2], "blks_per_sm:", candidate[0], "reg used:", candidate[1] * reg_per_thread * candidate[0] * get_kernel_info(kernel1)["blocksize"] + candidate[2] * reg_per_thread * candidate[0] * get_kernel_info(kernel2)["blocksize"], "smem used:", candidate[1] * get_kernel_info(kernel1)["shared_memory"] * candidate[0] + candidate[2] * get_kernel_info(kernel2)["shared_memory"] * candidate[0], "thread used:", candidate[1] * get_kernel_info(kernel1)["blocksize"] * candidate[0] + candidate[2] * get_kernel_info(kernel2)["blocksize"] * candidate[0])
    
#     print("-", kernel2, "as main kernel:")
#     candidates_ = fuse_kernel_info(kernel2, kernel1)
#     for candidate in candidates_:
#         if (candidate[2], candidate[1]) in generated_list: 
#             continue
#         blks_per_sm, ratio_2, ratio_1 = candidate[0], candidate[1], candidate[2]
#         print(f"// {kernel1}-{kernel2}-{ratio_1}-{ratio_2}", file=f)
#         mixed_kernel_code = generate_mixed_kernel([kernel_signatures[kernel1], kernel_signatures[kernel2]], blks_per_sm, [ratio_1, ratio_2])
#         print(mixed_kernel_code, file=f)

#     for candidate in candidates_:
#         if (candidate[2], candidate[1]) in generated_list:
#             continue
#         reg_per_thread = max(get_kernel_info(kernel1)["register"], get_kernel_info(kernel2)["register"])
#         print(f"-- {kernel1}_num:", candidate[2], f"{kernel2}_num:", candidate[1], "blks_per_sm:", candidate[0], "reg used:", candidate[2] * reg_per_thread * candidate[0] * get_kernel_info(kernel1)["blocksize"] + candidate[1] * reg_per_thread * candidate[0] * get_kernel_info(kernel2)["blocksize"], "smem used:", candidate[2] * get_kernel_info(kernel1)["shared_memory"] * candidate[0] + candidate[1] * get_kernel_info(kernel2)["shared_memory"] * candidate[0], "thread used:", candidate[2] * get_kernel_info(kernel1)["blocksize"] * candidate[0] + candidate[1] * get_kernel_info(kernel2)["blocksize"] * candidate[0])
    
#     f.close()
#     print('\n')

def gen_pair_code_iter(kernel1, kernel2)->list:
    # 判断文件是否存在
    # file_path = f"/home/jxdeng/workspace/tacker/ptb_kernels/mix_kernel/{kernel1}-{kernel2}.cu"
    # f = open(file_path, 'w')

    print("kernel pair: ", kernel1, kernel2)

    generated_list = [] # 去重

    candidates = fuse_kernel_info(kernel1, kernel2)
    for candidate in candidates:
        ret_code = ""
        blks_per_sm, ratio_1, ratio_2 = candidate[0], candidate[1], candidate[2]
        ret_code += f"// {kernel1}-{kernel2}-{ratio_1}-{ratio_2}\n"
        # print(f"// {kernel1}-{kernel2}-{ratio_1}-{ratio_2}", file=f)
        mixed_kernel_code = generate_mixed_kernel([kernel_signatures[kernel1], kernel_signatures[kernel2]], blks_per_sm, [ratio_1, ratio_2])
        ret_code += mixed_kernel_code
        # print(mixed_kernel_code, file=f)
        generated_list.append((ratio_1, ratio_2))
        yield ret_code, ratio_1, ratio_2, blks_per_sm

    candidates_ = fuse_kernel_info(kernel2, kernel1)
    for candidate in candidates_:
        if (candidate[2], candidate[1]) in generated_list: 
            continue
        ret_code = ""
        blks_per_sm, ratio_2, ratio_1 = candidate[0], candidate[1], candidate[2]
        # print(f"// {kernel1}-{kernel2}-{ratio_1}-{ratio_2}", file=f)
        ret_code += f"// {kernel1}-{kernel2}-{ratio_1}-{ratio_2}\n"
        mixed_kernel_code = generate_mixed_kernel([kernel_signatures[kernel1], kernel_signatures[kernel2]], blks_per_sm, [ratio_1, ratio_2])
        # print(mixed_kernel_code, file=f)
        ret_code += mixed_kernel_code
        yield ret_code, ratio_1, ratio_2, blks_per_sm


    print("-", kernel1, "as main kernel:")
    for candidate in candidates:
        reg_per_thread = max(get_kernel_info(kernel1)["register"], get_kernel_info(kernel2)["register"])
        print(f"-- {kernel1}_num:", candidate[1], f"{kernel2}_num:", candidate[2], "blks_per_sm:", candidate[0], "reg used:", candidate[1] * reg_per_thread * candidate[0] * get_kernel_info(kernel1)["blocksize"] + candidate[2] * reg_per_thread * candidate[0] * get_kernel_info(kernel2)["blocksize"], "smem used:", candidate[1] * get_kernel_info(kernel1)["shared_memory"] * candidate[0] + candidate[2] * get_kernel_info(kernel2)["shared_memory"] * candidate[0], "thread used:", candidate[1] * get_kernel_info(kernel1)["blocksize"] * candidate[0] + candidate[2] * get_kernel_info(kernel2)["blocksize"] * candidate[0])

    print("-", kernel2, "as main kernel:")
    for candidate in candidates_:
        if (candidate[2], candidate[1]) in generated_list:
            continue
        reg_per_thread = max(get_kernel_info(kernel1)["register"], get_kernel_info(kernel2)["register"])
        print(f"-- {kernel1}_num:", candidate[2], f"{kernel2}_num:", candidate[1], "blks_per_sm:", candidate[0], "reg used:", candidate[2] * reg_per_thread * candidate[0] * get_kernel_info(kernel1)["blocksize"] + candidate[1] * reg_per_thread * candidate[0] * get_kernel_info(kernel2)["blocksize"], "smem used:", candidate[2] * get_kernel_info(kernel1)["shared_memory"] * candidate[0] + candidate[1] * get_kernel_info(kernel2)["shared_memory"] * candidate[0], "thread used:", candidate[2] * get_kernel_info(kernel1)["blocksize"] * candidate[0] + candidate[1] * get_kernel_info(kernel2)["blocksize"] * candidate[0])

    # f.close()
    print('\n')
