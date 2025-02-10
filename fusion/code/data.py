'''
Author: diagonal
Date: 2023-11-16 15:07:33
LastEditors: diagonal
LastEditTime: 2023-11-30 11:36:13
FilePath: /tacker/mix_kernels/code/data.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''


# cp	39 	0 	128
# cutcp	42 	4096 	128
# fft	33 	2048 	128
# lbm	61 	0 	128
# mrif	32 	0 	256
# mriq	31 	0 	256
# sgemm	64 	512 	128
# tpacf	41 	13312 	256

SM_REGISTER_NUM = 65536
SM_THREAD_SLOT_NUM = 1024
SM_SHARED_MEMORY_NUM = 65536
kernel_info = {
    "cp" : {"register": 39, "shared_memory": 0, "blocksize": 128, "solo_ptb_blks": 8}, # 4 default, 8 close to ori(use for solo gptb), 6 for mix
    "cutcp": {"register": 42, "shared_memory": 4096, "blocksize": 128, "solo_ptb_blks": 6}, # 4 default, 6 close to ori
    "fft": {"register": 33, "shared_memory": 2048, "blocksize": 128, "solo_ptb_blks": 3},
    "lbm": {"register": 61, "shared_memory": 0, "blocksize": 128, "solo_ptb_blks": 1},
    "mrif": {"register": 32, "shared_memory": 0, "blocksize": 256, "solo_ptb_blks": 3},
    "mriq": {"register": 31, "shared_memory": 0, "blocksize": 256, "solo_ptb_blks": 4},
    "sgemm": {"register": 64, "shared_memory": 512, "blocksize": 128, "solo_ptb_blks": 4},
    "tpacf": {"register": 41, "shared_memory": 13312, "blocksize": 256, "solo_ptb_blks": 3},
    "stencil": {"register": 38, "shared_memory": 1024, "blocksize": 128, "solo_ptb_blks": 3}
}

def get_kernel_info(kernel_name):
    return kernel_info[kernel_name]

def gcd(a, b):
    if a < b:
        a, b = b, a
    while b != 0:
        a, b = b, a%b
    return a


def fuse_kernel_info(kernel1_name, kernel2_name)->list:
    # kernel1作为主kernel，保证kernel1再fused_kernel占用的thread总数 == solo_ptb_blks * blocksize
    kernel1_info = get_kernel_info(kernel1_name)
    kernel2_info = get_kernel_info(kernel2_name)
    kernel1_thread_num = kernel1_info["solo_ptb_blks"] * kernel1_info["blocksize"]
    candidates = []
    for blks_per_sm in range(1, 9):
        for kernel1_num in range(1, 9):
            if kernel1_num * kernel1_info["blocksize"] * blks_per_sm == kernel1_thread_num:
                for kernel2_num in range(1, 9):
                    if (kernel2_num * kernel2_info["blocksize"] * blks_per_sm <= SM_THREAD_SLOT_NUM - kernel1_thread_num) and (kernel1_info["register"] * kernel1_num * blks_per_sm * kernel1_info["blocksize"] + kernel2_info["register"] * kernel2_num * blks_per_sm * kernel2_info["blocksize"] <= SM_REGISTER_NUM) and (kernel1_info["shared_memory"] * kernel1_num * blks_per_sm + kernel2_info["shared_memory"] * kernel2_num * blks_per_sm <= SM_SHARED_MEMORY_NUM):
                        # 如果可化简，就化简
                        kernel1_num_ = int(kernel1_num / gcd(kernel1_num, kernel2_num))
                        kernel2_num_ = int(kernel2_num / gcd(kernel1_num, kernel2_num))
                        if ((blks_per_sm * gcd(kernel1_num, kernel2_num)), kernel1_num_, kernel2_num_) not in candidates:
                            candidates.append(((blks_per_sm * gcd(kernel1_num, kernel2_num)), kernel1_num_, kernel2_num_))
    
    # show candidates
    # print(f"candidates for {kernel1_name}_{kernel2_name}:")
    # for candidate in candidates:
    #     print("kernel1_num: ", candidate[1], "kernel2_num: ", candidate[2], "blks_per_sm: ", candidate[0], "reg used: ", candidate[1] * kernel1_info["register"] * candidate[0] * kernel1_info["blocksize"] + candidate[2] * kernel2_info["register"] * candidate[0] * kernel2_info["blocksize"], "sm used: ", candidate[1] * kernel1_info["shared_memory"] * candidate[0] + candidate[2] * kernel2_info["shared_memory"] * candidate[0], "thread used: ", candidate[1] * kernel1_info["blocksize"] * candidate[0] + candidate[2] * kernel2_info["blocksize"] * candidate[0])
    return candidates