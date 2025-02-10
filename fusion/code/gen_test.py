'''
Author: diagonal
Date: 2023-11-17 16:41:25
LastEditors: diagonal
LastEditTime: 2023-12-05 21:54:49
FilePath: /tacker/mix_kernels/code/gen_test.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
from common_code import common_header, time_event_create_code, main_func_begin_code, main_func_end_code
from cp_code import get_cp_header_code, get_cp_code_before_mix_kernel, get_cp_code_after_mix_kernel, cp_gptb_params_list, cp_gptb_params_list_new
from cutcp_code import get_cutcp_header_code, get_cutcp_code_before_mix_kernel, get_cutcp_code_after_mix_kernel, cutcp_gptb_params_list, cutcp_gptb_params_list_new
from fft_code import get_fft_header_code, get_fft_code_before_mix_kernel, get_fft_code_after_mix_kernel, fft_gptb_params_list, fft_gptb_params_list_new
from lbm_code import get_lbm_header_code, get_lbm_code_before_mix_kernel, get_lbm_code_after_mix_kernel, lbm_gptb_params_list, lbm_gptb_params_list_new
from mrif_code import get_mrif_header_code, get_mrif_code_before_mix_kernel, get_mrif_code_after_mix_kernel, mrif_gptb_params_list, mrif_gptb_params_list_new
from mriq_code import get_mriq_header_code, get_mriq_code_before_mix_kernel, get_mriq_code_after_mix_kernel, mriq_gptb_params_list, mriq_gptb_params_list_new
from sgemm_code import get_sgemm_header_code, get_sgemm_code_before_mix_kernel, get_sgemm_code_after_mix_kernel, sgemm_gptb_params_list, sgemm_gptb_params_list_new
from tpacf_code import get_tpacf_header_code, get_tpacf_code_before_mix_kernel, get_tpacf_code_after_mix_kernel, tpacf_gptb_params_list
from stencil_code import get_stencil_header_code, get_stencil_code_before_mix_kernel, get_stencil_code_after_mix_kernel, stencil_gptb_params_list, stencil_gptb_params_list_new
from data import get_kernel_info, fuse_kernel_info
# 如果没有对应函数则留一个匿名空函数
func_dict = {
    "cp": {"header": get_cp_header_code, "before": get_cp_code_before_mix_kernel, "after": get_cp_code_after_mix_kernel},
    "cutcp": {"header": get_cutcp_header_code, "before": get_cutcp_code_before_mix_kernel, "after": get_cutcp_code_after_mix_kernel},
    "fft": {"header": get_fft_header_code, "before": get_fft_code_before_mix_kernel, "after": get_fft_code_after_mix_kernel},
    "lbm": {"header": get_lbm_header_code, "before": get_lbm_code_before_mix_kernel, "after": get_lbm_code_after_mix_kernel},
    "mrif": {"header": get_mrif_header_code, "before": get_mrif_code_before_mix_kernel, "after": get_mrif_code_after_mix_kernel},
    "mriq": {"header": get_mriq_header_code, "before": get_mriq_code_before_mix_kernel, "after": get_mriq_code_after_mix_kernel},
    "sgemm": {"header": get_sgemm_header_code, "before": get_sgemm_code_before_mix_kernel, "after": get_sgemm_code_after_mix_kernel},
    # "tpacf": {"header": get_tpacf_header_code, "before": get_tpacf_code_before_mix_kernel, "after": get_tpacf_code_after_mix_kernel}
    "stencil": {"header": get_stencil_header_code, "before": get_stencil_code_before_mix_kernel, "after": get_stencil_code_after_mix_kernel}
}

param_dict = {
    "cp": cp_gptb_params_list,
    "cutcp": cutcp_gptb_params_list,
    "fft": fft_gptb_params_list,
    "lbm": lbm_gptb_params_list,
    "mrif": mrif_gptb_params_list,
    "mriq": mriq_gptb_params_list,
    "sgemm": sgemm_gptb_params_list,
    # "tpacf": tpacf_gptb_params_list
    "stencil": stencil_gptb_params_list
}

param_dict_new = {
    "cp": cp_gptb_params_list_new,
    "cutcp": cutcp_gptb_params_list_new,
    "fft": fft_gptb_params_list_new,
    "lbm": lbm_gptb_params_list_new,
    "mrif": mrif_gptb_params_list_new,
    "mriq": mriq_gptb_params_list_new,
    "sgemm": sgemm_gptb_params_list_new,
    "stencil": stencil_gptb_params_list_new
}

mix_kernel_code = """
  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3({kernel_grid_x}, 1, 1);
        dim3 mix_kernel_block = dim3({kernel_block_x}, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_{kernel1}_{kernel2}_kernel_{ratio_1}_{ratio_2} <<<mix_kernel_grid, mix_kernel_block>>>({kernel_args})));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] {kernel1}_{kernel2} {ratio_1}_{ratio_2} took %f ms\\n\\n", kernel_time);
  // ---------------------------------------------------------------------------------------

"""

gptb_kernel_code = """
    std::vector<float> time_vec;
    // GPTB
    // ---------------------------------------------------------------------------------------
        dim3 gptb_kernel_grid = dim3({kernel_grid_x}, 1, 1);
        dim3 gptb_kernel_block = dim3({kernel_block_x}, 1, 1);
        for(int i = 0; i < 30; ++i) {{
            cudaErrCheck(cudaEventRecord(startKERNEL));
            checkKernelErrors((g_general_ptb_{kernel} <<<gptb_kernel_grid, gptb_kernel_block>>>({kernel_args})));
            cudaErrCheck(cudaEventRecord(stopKERNEL));
            cudaErrCheck(cudaEventSynchronize(stopKERNEL));
            cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
        }}

        // sort & get average
        std::sort(time_vec.begin(), time_vec.end());
        float gptb_{kernel}_time = 0.0f;
        for(int i = 10; i < 20; ++i) {{
            gptb_{kernel}_time += time_vec[i];
        }}
        gptb_{kernel}_time /= 10.0f;
        time_vec.clear();
        printf("[GPTB] {kernel} took %f ms\\n", gptb_{kernel}_time);
        printf("[GPTB] {kernel} blks: %d\\n\\n", end_blk_no - start_blk_no);

        printf("---------------------------\\n");
"""



def gen_fused_code(kernel1, kernel2, kernel_grid_size, kernel_block_size, ratio):
    full_code = ""
    full_code += common_header
    full_code += func_dict[kernel1]["header"]() + func_dict[kernel2]["header"]()
    full_code += f"\n#include \"mix_kernel/{kernel1}_{kernel2}_{ratio[0]}_{ratio[1]}.cu\" \n"
    full_code += main_func_begin_code
    full_code += time_event_create_code
    full_code += func_dict[kernel1]["before"]()
    full_code += func_dict[kernel2]["before"]()
    full_code += mix_kernel_code.format(kernel1=kernel1, kernel2=kernel2, kernel_args=param_dict[kernel1] + ", " + param_dict[kernel2], kernel_grid_x=kernel_grid_size, kernel_block_x=kernel_block_size, ratio_1=ratio[0], ratio_2=ratio[1])
    full_code += func_dict[kernel1]["after"]()
    full_code += func_dict[kernel2]["after"]()
    full_code += main_func_end_code
    return full_code

def gen_gptb_code(kernel, kernel_grid_size, kernel_block_size, start_blk_no, end_blk_no):
    full_code = ""
    full_code += common_header
    full_code += func_dict[kernel]["header"]()
    # full_code += f"\n#include \"mix_kernel/{kernel}.cu\" \n"
    full_code += main_func_begin_code
    full_code += time_event_create_code
    full_code += func_dict[kernel]["before"]()
    full_code += gptb_kernel_code.format(kernel_args=param_dict_new[kernel], kernel=kernel, kernel_grid_x=kernel_grid_size, kernel_block_x=kernel_block_size)
    full_code = full_code.replace("start_blk_no", str(start_blk_no))
    full_code = full_code.replace("end_blk_no", str(end_blk_no))
    full_code += main_func_end_code
    return full_code

def gen_test_code_1_1(kernel1, kernel2, kernel_grid_x):
    full_code = ""
    full_code += common_header
    full_code += func_dict[kernel1]["header"]() + func_dict[kernel2]["header"]()
    full_code += f"\n#include \"mix_kernel/{kernel1}-{kernel2}.cu\" \n"
    full_code += main_func_begin_code
    full_code += time_event_create_code
    full_code += func_dict[kernel1]["before"]()
    full_code += func_dict[kernel2]["before"]()
    full_code += mix_kernel_code.format(kernel1=kernel1, kernel2=kernel2, kernel_args=param_dict[kernel1] + ", " + param_dict[kernel2], kernel_grid_x=kernel_grid_x, kernel_block_x=get_kernel_info(kernel1)["blocksize"] + get_kernel_info(kernel2)["blocksize"])
    full_code += func_dict[kernel1]["after"]()
    full_code += func_dict[kernel2]["after"]()
    full_code += main_func_end_code
    return full_code

SM_NUM = 68
# test
if __name__ == "__main__":
    # print(gen_test_code_1_1("cp", "tpacf", SM_NUM * max(get_kernel_info("cp")["solo_ptb_blks"], get_kernel_info("tpacf")["solo_ptb_blks"])))
    file_dir = "../../ptb_kernels/mix/"
    kernel_pairs = []
    for i, kernel1 in enumerate(func_dict):
        for j, kernel2 in enumerate(func_dict):
            if i < j:
                kernel_pairs.append([kernel1, kernel2])
    # 两两生成mix_kernel
    for kernel1, kernel2 in kernel_pairs:
        candidates = fuse_kernel_info(kernel1, kernel2)
        for candidate in candidates:
            with open(file_dir + f"{kernel1}_{kernel2}_{candidate[1]}_{candidate[2]}.cu", "w") as f:
                f.write(gen_fused_code(kernel1, kernel2, SM_NUM * candidate[0], candidate[1] * get_kernel_info(kernel1)["blocksize"] + candidate[2] * get_kernel_info(kernel2)["blocksize"], (candidate[1], candidate[2])))
        candidates_ = fuse_kernel_info(kernel2, kernel1)
        for candidate in candidates_:
            if (candidate[0], candidate[2], candidate[1]) in candidates:
                continue
            with open(file_dir + f"{kernel1}_{kernel2}_{candidate[2]}_{candidate[1]}.cu", "w") as f:
                f.write(gen_fused_code(kernel1, kernel2, SM_NUM * candidate[0], candidate[2] * get_kernel_info(kernel1)["blocksize"] + candidate[1] * get_kernel_info(kernel2)["blocksize"], (candidate[2], candidate[1])))

        # with open(file_dir + f"{kernel1}_{kernel2}.cu", "w") as f:
        #     f.write(gen_test_code_1_1(kernel1, kernel2, SM_NUM * max(get_kernel_info(kernel1)["solo_ptb_blks"], get_kernel_info(kernel2)["solo_ptb_blks"])))


