'''
Author: diagonal
Date: 2023-11-19 13:04:17
LastEditors: diagonal
LastEditTime: 2023-12-07 11:48:23
FilePath: /tacker/mix_kernels/run/run.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
import glob
import subprocess
import re
import sys
import pandas as pd

k_list = ["cp", "cutcp", "fft", "lbm", "mrif", "mriq", "sgemm", "stencil"]

data = {'pair':[], 'fusion_info': [], 'ori1': [], 'ptb1': [], 'ori2': [], 'ptb2': [], 'mix': []}

num_repetitions = 50

warning_file = open("warning.log", "w")

def find_mix_executables(kernel1, kernel2):
    pattern = f'{kernel1}_{kernel2}_[0-9]_[0-9]_mix'  # 匹配数字_数字_mix

    # 使用 glob 模块查找匹配的文件
    mix_files = glob.glob(pattern)

    return mix_files

def run_and_collect_info(kernel1, kernel2):
    # 查找当前文件夹下的所有以kernel1_kernel2开头，mix结尾的可执行文件
    exe_list = find_mix_executables(kernel1, kernel2)
    print(f"exe_list: {exe_list}")

    for executable_name in exe_list:
        command = f"./{executable_name}"

        # 从executable_name中提取ratio
        ratio_pattern = re.compile(r'(\d+)_(\d+)_mix')
        ratio_matches = ratio_pattern.findall(executable_name)
        if ratio_matches:
            ratio_1, ratio_2 = ratio_matches[0]
            data['pair'].append(f"{kernel1}_{kernel2}")
            data['fusion_info'].append(f"{ratio_1}:{ratio_2}")
        else:
            print(f"Error running {executable_name}, can't find ratio, Exit!")
            exit(1)

        print(f"command: {command}")

        try:
            # warmup
            exit_flag = False
            for i_ in range(5):
                print(f"warmup {i_+1}/5")
                output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT, timeout=5)
                if "too many resources" in output:
                    print(f"Error running {executable_name}, Error: ---\n{output}\n---\n, Exit!", file=sys.stderr)
                    exit_flag = True
                    # 使用kill命令杀死进程
                    subprocess.run(f"pkill -f {executable_name}", shell=True)
                    break
                # elif "error" in output or "Error" in output:
                #     print(f"Error running {executable_name}, Error: ---\n{output}\n---\n", file=sys.stderr)
                #     input("Press Enter to continue...")
            
            if exit_flag:
                data['ori1'].append(0)
                data['ori2'].append(0)
                data['ptb1'].append(0)
                data['ptb2'].append(0)
                data['mix'].append(0)

                continue

            data_ori1 = []
            data_ori2 = []
            data_ptb1 = []
            data_ptb2 = []
            data_mix = []
            for i_ in range(num_repetitions):
                if i_ % 10 == 9:
                    print(f"run {i_+1}/{num_repetitions}")
                output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
                # if "Error" in output:
                #     print(f"Error running {executable_name}, Error: ---\n{output}\n---\n, Exit!", file=sys.stderr)
                #     exit(1)
                # 提取运行时间的数字部分
                pattern = re.compile(r'\[(\w+)\] (.+?) took (\d+\.\d+) ms')
                runtime_matches = pattern.findall(output)
                if runtime_matches:
                    ori_idx = 1
                    ptb_idx = 1
                    for match in runtime_matches:
                        group, _, time = match
                        if group == 'ORI' and ori_idx == 1:
                            data_ori1.append(float(time))
                            ori_idx += 1
                        elif group == 'ORI' and ori_idx == 2:
                            data_ori2.append(float(time))
                            ori_idx += 1
                        elif group == 'PTB' and ptb_idx == 1:
                            data_ptb1.append(float(time))
                            ptb_idx += 1
                        elif group == 'PTB' and ptb_idx == 2:
                            data_ptb2.append(float(time))
                            ptb_idx += 1
                        elif group == 'MIX':
                            data_mix.append(float(time))
            
            # 处理数据
            data_ori1.sort()
            data_ori2.sort()
            data_ptb1.sort()
            data_ptb2.sort()
            data_mix.sort()

            if data_ori1.__len__() != num_repetitions or data_ori2.__len__() != num_repetitions or data_ptb1.__len__() != num_repetitions or data_ptb2.__len__() != num_repetitions or data_mix.__len__() != num_repetitions:
                print(f"Warning running {executable_name}, data length not equal to num_repetitions")
                print(f"data_ori1: {data_ori1.__len__()}, data_ori2: {data_ori2.__len__()}, data_ptb1: {data_ptb1.__len__()}, data_ptb2: {data_ptb2.__len__()}, data_mix: {data_mix.__len__()}")

            start_index = int(0.125 * num_repetitions)
            end_index = int(0.875 * num_repetitions)
            if len(data_ori1) > 0:
                data['ori1'].append(sum(data_ori1[start_index:end_index]) / (end_index - start_index))
            else:
                data['ori1'].append(0)
            if len(data_ori2) > 0:
                data['ori2'].append(sum(data_ori2[start_index:end_index]) / (end_index - start_index))
            else:
                data['ori2'].append(0)
            if len(data_ptb1) > 0:
                data['ptb1'].append(sum(data_ptb1[start_index:end_index]) / (end_index - start_index))
            else:
                data['ptb1'].append(0)
            if len(data_ptb2) > 0:
                data['ptb2'].append(sum(data_ptb2[start_index:end_index]) / (end_index - start_index))
            else:
                data['ptb2'].append(0)
            if len(data_mix) > 0:
                data['mix'].append(sum(data_mix[start_index:end_index]) / (end_index - start_index))
            else:
                data['mix'].append(0)

            if len(data_ori1) > 0 and abs(data_ori1[start_index] - data_ori1[end_index - 1]) > 0.1:
                print(f"Warning: {executable_name} ORI1 has large variance -- {data_ori1[start_index]} - {data_ori1[end_index - 1]}", file=warning_file)
            if len(data_ori2) > 0 and abs(data_ori2[start_index] - data_ori2[end_index - 1]) > 0.1:
                print(f"Warning: {executable_name} ORI2 has large variance -- {data_ori2[start_index]} - {data_ori2[end_index - 1]}", file=warning_file)
            if len(data_ptb1) > 0 and abs(data_ptb1[start_index] - data_ptb1[end_index - 1]) > 0.1:
                print(f"Warning: {executable_name} PTB1 has large variance -- {data_ptb1[start_index]} - {data_ptb1[end_index - 1]}", file=warning_file)
            if len(data_ptb2) > 0 and abs(data_ptb2[start_index] - data_ptb2[end_index - 1]) > 0.1:
                print(f"Warning: {executable_name} PTB2 has large variance -- {data_ptb2[start_index]} - {data_ptb2[end_index - 1]}", file=warning_file)
            if len(data_mix) > 0 and abs(data_mix[start_index] - data_mix[end_index - 1]) > 0.1:
                print(f"Warning: {executable_name} MIX has large variance -- {data_mix[start_index]} - {data_mix[end_index - 1]}", file=warning_file)


        except subprocess.CalledProcessError:
            print(f"Error running {executable_name}")
            subprocess.run(f"pkill -f {executable_name}", shell=True)
            data['ori1'].append(0)
            data['ori2'].append(0)
            data['ptb1'].append(0)
            data['ptb2'].append(0)
            data['mix'].append(0)
            input("Press Enter to continue...")
            continue
        except subprocess.TimeoutExpired:
            print(f"Error running {executable_name}, TimeoutExpired")
            subprocess.run(f"pkill -f {executable_name}", shell=True)
            data['ori1'].append(0)
            data['ori2'].append(0)
            data['ptb1'].append(0)
            data['ptb2'].append(0)
            data['mix'].append(0)
            input("Press Enter to continue...")
            continue


def my_exit():
    print("Exit!")
    warning_file.close()
    # 创建一个DataFrame
    df = pd.DataFrame(data)

    # 将DataFrame写入Excel文件
    df.to_excel('output_new.xlsx', index=False)
    print(data)

if __name__ == "__main__":
    kernel_pairs = []
    for i, kernel1 in enumerate(k_list):
        for j, kernel2 in enumerate(k_list):
            if i < j:
                if "stencil" in kernel1 or "stencil" in kernel2:
                    kernel_pairs.append([kernel1, kernel2])

    print("len(kernel_pairs): ", len(kernel_pairs))
    print("kernel_pairs: ", kernel_pairs)

    import atexit
    atexit.register(my_exit)

    for pair in kernel_pairs:
        kernel1, kernel2 = pair
        run_and_collect_info(kernel1, kernel2)