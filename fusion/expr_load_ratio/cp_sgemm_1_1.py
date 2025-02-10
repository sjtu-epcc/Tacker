import re
import subprocess
import sys

compile_command = "make cp_sgemm_mix -j $(nproc)"
command = "./cp_sgemm_1_1_mix"

def compile(load_ratio, mix_sgemm_task_blk_num):
    # open cp_sgemm_1_1.cu, replace mix_mrif_task_blk_num
    with open('cp_sgemm_1_1.cu', 'r') as f:
        content = f.read()
        content = re.sub(r'int mix_sgemm_task_blk_num = \d+;', f'int mix_sgemm_task_blk_num = {mix_sgemm_task_blk_num};', content)
        content = re.sub(r'float load_ratio = \d+\.\d+;', f'float load_ratio = {load_ratio};', content)
    
    # 将修改后的内容写回文件
    with open('cp_sgemm_1_1.cu', 'w') as f:
        f.write(content)

    # 编译
    try:
        output = subprocess.check_output(compile_command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error compile, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
        exit(1)

def run():
    # 运行
    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error running {command}, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
        exit(1)
    print(output)
            

if __name__ == "__main__":
    max_cp_blks = 320 * 512
    max_sgemm_blks = 774
    load_ratios = [0.35, 0.72, 1.06, 1.41]
    for load_ratio in load_ratios:
        for i in range(1, 1024 + 1):
            if i % 68 == 0:
                print(f"--- {load_ratio} ---")
                compile(load_ratio, i)
                run()
        
        # input("Press Enter to continue...")
