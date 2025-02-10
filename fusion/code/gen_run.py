'''
Author: diagonal
Date: 2023-12-05 21:43:33
LastEditors: diagonal
LastEditTime: 2023-12-07 11:30:27
FilePath: /tacker/mix_kernels/code/gen_run.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
import os
import re
import sys
import subproces

from data import get_kernel_info, fuse_kernel_info
from util import extract_kernel_signature, process_parameter_list

kernel_list = ['cp', 'cutcp', 'fft', 'lbm', 'mrif', 'mriq', 'sgemm', 'stencil']

from gen_mix import gen_pair_code_iter
from gen_test import gen_fused_code


SM_NUM = 68
def gen_code_file(kernel1, kernel2):
    kernel_file_dir = "../mix_kernel/"
    run_file_dir = "../run/"
    # 两两生成mix_kernel
    mix_kernel_code_iter = gen_pair_code_iter(kernel1=kernel1, kernel2=kernel2)
    # 迭代
    for mix_kernel_code, ratio1, ratio2, blks_per_sm in mix_kernel_code_iter:
        exit_flag = False

        # check if run_file_dir/xx_xx_mix exists
        # if os.path.exists(run_file_dir + f"{kernel1}_{kesnel2}_{ratio1}_{ratio2}_mix"):
        #     print(f"[skip {kernel1}_{kernel2}_{ratio1}_{ratio2}_mix]")
        #     continue

        
        print(f"\nGen {kernel1}_{kernel2}_{ratio1}_{ratio2}.cu, blks_per_sm: {blks_per_sm}")
        with open(kernel_file_dir + f"{kernel1}_{kernel2}_{ratio1}_{ratio2}.cu", "w") as f:
            f.write(mix_kernel_code)
        
        run_code = gen_fused_code(kernel1, kernel2, SM_NUM * blks_per_sm, get_kernel_info(kernel1)["blocksize"] * ratio1 + get_kernel_info(kernel2)["blocksize"] * ratio2, (ratio1, ratio2))

        with open(run_file_dir + f"{kernel1}_{kernel2}_{ratio1}_{ratio2}.cu", "w") as f: 
            f.write(run_code)
        # 编译前使用子进程修改../kernel下的{kernel1}_kernel.cu和{kernel2}_kernel.cu，替换代码中的asm指令参数
        # asm volatile("bar.sync %0, %1;" : : "r"(num1), "r"(num2) : "memory"); 
        
        sync_num = 0
        func_no = 0
        with open(f"../kernel/{kernel1}_kernel.cu", "r") as f:
            lines = f.readlines()
        # 使用正则表达式匹配并替换num1的位置
        pattern = r'\s*asm volatile\("bar\.sync %0, %1;" : : "r"\((\d+)\), "r"\((\d+)\) : "memory"\);'
        func_start = False
        dif_lines = []
        func_end = False
        func_match = False
        stop_flag = False
        for i, line in enumerate(lines):
            if line.startswith("__device__ void general_ptb_") or line.startswith("__device__ void G_"):
                func_start = True
                func_end = False
                func_match = False
            if line.startswith("}") and func_start == True:
                func_end = True
                if func_match:
                    sync_num += 1
                    func_no += 1
                func_start = False
                if func_no >= int(ratio1):
                    stop_flag = True
            if stop_flag:
                break
            match = re.match(pattern, line)
            if match and func_start and not func_end:
                # print(f"kernel {kernel1} match, origin line: {line}")
                old_num2 = match.group(2)
                new_line = f'asm volatile("bar.sync %0, %1;" : : "r"({sync_num + 1}), "r"({old_num2}) : "memory");\n'
                dif_lines.append((i, line, new_line))
                # print(f'File "{kernel1}_kernel.cu", Line {i + 1}: \n Before \n--{line.strip()}, After \n--{new_line}')
                lines[i] = new_line
                func_match = True
        # 写回文件
        with open(f"../kernel/{kernel1}_kernel.cu",'w') as file:
            file.write(''.join(lines))
        for i, old_line, new_line in dif_lines:
            print(f'File "{kernel1}_kernel.cu", Line {i + 1}: \n--{old_line.strip()}  -->\n--{new_line}', end="")

        with open(f"../kernel/{kernel2}_kernel.cu", "r") as f:
            lines = f.readlines()
        func_no = 0
        # 使用正则表达式匹配并替换num1的位置
        dif_lines = []
        func_start = False
        func_end = False
        func_match = False
        stop_flag = False   
        for i, line in enumerate(lines):
            if line.startswith("__device__ void general_ptb_") or line.startswith("__device__ void G_"):
                func_start = True
                func_end = False
                func_match = False
            if line.startswith("}") and func_start == True:
                func_end = True
                if func_match:
                    sync_num += 1
                    func_no += 1
                func_start = False
                if func_no >= int(ratio2):
                    stop_flag = True
            if stop_flag:
                break
            match = re.match(pattern, line)
            if match and func_start and not func_end:
                # print(f"kernel {kernel2} match, origin line: {line}")
                old_num2 = match.group(2)
                new_line = f'asm volatile("bar.sync %0, %1;" : : "r"({sync_num + 1}), "r"({old_num2}) : "memory");\n'
                dif_lines.append((i, line, new_line))
                # print(f'File "{kernel2}_kernel.cu", Line {i + 1}: \n Before \n--{line.strip()}, After \n--{new_line}')
                lines[i] = new_line
                func_match = True
        # 写回文件
        with open(f"../kernel/{kernel2}_kernel.cu",'w') as file:
            file.write(''.join(lines))
        for i, old_line, new_line in dif_lines:
            print(f'File "{kernel2}_kernel.cu", Line {i + 1}: \n--{old_line.strip()}  -->\n--{new_line}', end="")
        
        print("Gen code success!")

        # 利用makefile编译，makefile在../run/下，命令为`make {kernel1}_{kernel2}_mix`
        cmd = f"make {kernel1}_{kernel2}_mix"
        # 使用子进程执行命令
        print(f"Compile {kernel1}_{kernel2}_{ratio1}_{ratio2}...")
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, cwd=run_file_dir)
        if "Error" in output or "errors" in output:
            print(f"Error compile {kernel1}_{kernel2}_{ratio1}_{ratio2}, Error: ---\n{output}\n---\n", file=sys.stderr)
            exit_flag = True
        
        if exit_flag:
            print("Exit because of compile error!")
            exit(1)
        print("Compile success!")

        # check 是否编译成功
        cmd_output = subprocess.check_output(f"ls | grep {kernel1}_{kernel2}_{ratio1}_{ratio2}", shell=True, text=True, stderr=subprocess.STDOUT, cwd=run_file_dir)
        print("Check compile result...")
        if f"{kernel1}_{kernel2}_{ratio1}_{ratio2}_mix" in cmd_output:
            print(f"Success compile {kernel1}_{kernel2}_{ratio1}_{ratio2}!")
        else:
            print(f"Error compile {kernel1}_{kernel2}_{ratio1}_{ratio2}: ", file=sys.stderr)
            print(cmd_output)
            exit_flag = True
        
        if exit_flag:
            print("Exit because of compile error!")
            exit(1)
        print("Success check compile result!")
    
        # test run
        cmd = f"./{kernel1}_{kernel2}_{ratio1}_{ratio2}_mix"
        # 使用子进程执行命令
        print(f"Run {kernel1}_{kernel2}_{ratio1}_{ratio2}...")

        try:
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, cwd=run_file_dir, timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Error TimeoutExpired running {cmd}, timout!", file=sys.stderr)
            # 使用杀死进程
            subprocess.run(f"pkill -f {cmd}", shell=True)
            exit_flag = True
        except subprocess.CalledProcessError as e:
            # too many resources
            print(f"Error CalledProcessError running {cmd}, Error: ---\n{e.output + str(e.stderr) + str(e.stdout)}\n---\n")
            if "Aborted" in e.output:
                input("[WARNING]check file and press Enter to continue...")
                # 使用kill命令杀死进程
                subprocess.run(f"pkill -f {cmd}", shell=True)
            else:
                exit_flag = True
                # 使用kill命令杀死进程
                subprocess.run(f"pkill -f {cmd}", shell=True)
        except Exception as e:
            print(f"Error Exception running {cmd}, Error: ---\n{e}\n---\n", file=sys.stderr)
            # 使用kill命令杀死进程
            subprocess.run(f"pkill -f {cmd}", shell=True)
            exit_flag = True
        
        if "Error" in output or "errors" in output:
            print(f"Error running {cmd}, Error: ---\n{output}\n---\n", file=sys.stderr)
            # 使用kill命令杀死进程
            subprocess.run(f"pkill -f {cmd}", shell=True)
            exit_flag = True
        
        if exit_flag:
            print("Exit because of running error!")
            file_path = os.path.join(run_file_dir, f"{kernel1}_{kernel2}_{ratio1}_{ratio2}_mix")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
            exit(1)
            
        # 删除run_file_dir + f"{kernel1}_{kernel2}_{ratio1}_{ratio2}.cu"
        file_path = os.path.join(run_file_dir, f"{kernel1}_{kernel2}_{ratio1}_{ratio2}.cu")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found")
            input("[WARNING]check file and press Enter to continue...")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            exit_flag = True
        
        if exit_flag:
            print("Exit because of deleting error!")
            exit(1)

        print("Run success!")

if __name__ == "__main__":
    kernel_pairs = []
    for i, kernel1 in enumerate(kernel_list):
        for j, kernel2 in enumerate(kernel_list):
            if i < j:
                kernel_pairs.append([kernel1, kernel2])
    # 两两生成mix_kernel
    for kernel1, kernel2 in kernel_pairs:
        gen_code_file(kernel1, kernel2)