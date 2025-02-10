import os
import re
import subprocess
import sys

from data import get_kernel_info, fuse_kernel_info
from util import extract_kernel_signature, process_parameter_list

kernel_list = ['cp', 'cutcp', 'fft', 'lbm', 'mrif', 'mriq', 'sgemm', 'stencil']

from gen_mix import gen_pair_code_iter
from gen_test import gen_gptb_code


SM_NUM = 68
run_file_dir = "../gptb_profile/"
def gen_gptb_code_file(kernel, task_blk_num, profile_log_file):
    print("---kernel: ", kernel, "task_blk_num: ", task_blk_num, "---")
    
    run_code = gen_gptb_code(kernel, SM_NUM * get_kernel_info(kernel)["solo_ptb_blks"], get_kernel_info(kernel)["blocksize"], 0, task_blk_num)    
    print("Gen code success!")

    with open(os.path.join(run_file_dir, f"{kernel}_gptb_profile.cu"), "w") as f:
        f.write(run_code)

    exit_flag = False

    # 利用makefile编译，makefile在../run/下，命令为`make {kernel1}_{kernel2}_mix`
    cmd = f"make {kernel}"
    # 使用子进程执行命令
    print(f"Compile {kernel}...")
    output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, cwd=run_file_dir)
    if "Error" in output:
        print(f"Error compile {kernel}, Error: ---\n{output}\n---\n", file=sys.stderr)
        exit_flag = True
    
    if exit_flag:
        print("Exit because of compile error!")
        exit(1)

    # check 是否编译成功
    cmd_output = subprocess.check_output(f"ls | grep {kernel}_out", shell=True, text=True, stderr=subprocess.STDOUT, cwd=run_file_dir)
    print("Check compile result...")
    if f"{kernel}_out" in cmd_output:
        print(f"Success compile {kernel}!")
    else:
        print(f"Error compile {kernel}: ", file=sys.stderr)
        print(cmd_output)
        exit_flag = True
    
    if exit_flag:
        print("Exit because of compile error!")
        exit(1)
    print("Success check compile result!")

    # test run
    cmd = f"./{kernel}_out"
    # 使用子进程执行命令
    print(f"Run {kernel}_out...")

    try:
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, cwd=run_file_dir, timeout=10)
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
        file_path = os.path.join(run_file_dir, f"{kernel}_out")
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
        exit(1)
    print(output)
    print(output, file=profile_log_file, flush=True)

    print("Run success!")

if __name__ == "__main__":
    # 读取json文件
    import json
    with open("./kinfo.json", "r") as f:
        kinfo = json.load(f)

    for i, kernel in enumerate(kernel_list):
        # 打开{kernel}_profile.log文件
        with open(os.path.join(run_file_dir, f"./{kernel}_gptb_profile.log"), "w") as log_file:
            for task_blk_num in range(SM_NUM, kinfo[f"{kernel}"]["ori_blks"] + 1):
                if task_blk_num % SM_NUM == 0:
                    gen_gptb_code_file(kernel, task_blk_num, log_file)