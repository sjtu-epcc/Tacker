'''
Author: diagonal
Date: 2023-11-18 13:22:06
LastEditors: diagonal
LastEditTime: 2023-11-30 11:09:19
FilePath: /tacker/mix_kernels/code/makefile.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
import re
import itertools

def extract_compile_flags(command):
    # 按照空格切分命令，提取编译参数
    flags = command.split(" ")[3:-1]
    print(f"Flags: {flags}")
    return flags

def merge_flags(flags1: list, flags2: list):
    # 合并两个参数的去重集合
    flags = set(flags1 + flags2)
    # 如果包含"-dlcm=ca"这个参数，则放到最后
    if "-dlcm=ca" in flags:
        flags.remove("-dlcm=ca")
        flags.remove("-Xptxas")
        flags = list(flags)
        flags.append("-Xptxas")
        flags.append("-dlcm=ca")
    return " ".join(flags)

def generate_makefile(source_makefile):
    with open(source_makefile, 'r') as file:
        makefile_content = file.read()

    # 提取每个命令对应的编译参数
    command_flags = {}
    pattern = re.compile(r'(\b(?:cp|cutcp|fft|lbm|mrif|mriq|sgemm|stencil)\b):\s*([\s\S]+?)(?=\w+:|$)')
    matches = pattern.findall(makefile_content)
    for target, command in matches:
        print(f"Extracting flags from {target}")
        print(f"Command: {command}")
        command_flags[target] = extract_compile_flags(command)

    new_content = ""
    # 生成两两组合的新命令
    combinations = list(itertools.combinations(command_flags.keys(), 2))
    for cmd1, cmd2 in combinations:
        merged_flags = merge_flags(command_flags[cmd1], command_flags[cmd2])
        new_command = f"nvcc -o {cmd1}_{cmd2}_mix -I../ {merged_flags} {cmd1}_{cmd2}.cu"
        new_content += f"{cmd1}_{cmd2}_mix: \n\t{new_command}\n\n"

    # 输出合并后的Makefile
    with open('Makefile_merged_new', 'w') as file:
        file.write(new_content)

if __name__ == "__main__":
    generate_makefile('Makefile')  # 输入你的Makefile文件名
