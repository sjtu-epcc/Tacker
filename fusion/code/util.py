'''
Author: diagonal
Date: 2023-11-15 23:13:55
LastEditors: diagonal
LastEditTime: 2023-11-16 22:06:50
FilePath: /tacker/mix_kernels/code/util.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
import re

def extract_kernel_signature(kernel_code):
    # 使用正则表达式匹配 CUDA kernel 函数签名
    pattern = re.compile(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)')
    match = pattern.search(kernel_code)

    if match:
        # 如果匹配成功，返回函数签名的两个组成部分：函数名和参数
        function_name = match.group(1)
        function_name = function_name.split('_')[-1].replace(' ', '')

        function_params = match.group(2).strip()

        return function_name, function_params
    else:
        # 如果匹配失败，输出错误信息，退出进程
        print("Failed to extract kernel signature.")
        exit(1)

def process_parameter_list(name, id, parameter_list):
    # 使用正则表达式匹配数据类型和参数名，支持指针
    pattern = re.compile(r'\b(\w+(?:\s*\*\s*\w*)?)\s+(\w+)\b')
    matches = pattern.findall(parameter_list)

    # 将matches中每个tuple的第一个元素加上name
    for i in range(len(matches)):
        matches[i] = (matches[i][0], name + str(id) + '_' + matches[i][1])

    # 返回一个 List(tuple(str, str))
    return matches