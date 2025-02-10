import re
import subprocess
import sys

compile_command = "make fft_mrif_mix -j $(nproc)"
command = "./fft_mrif_3_2_mix"

best_improve = -11111.0

def compile(mix_mrif_task_blk_num):
    # 打开fft_mrif_3_2.cu文件，替换`int mix_mrif_task_blk_num = `为指定的参数
    with open('fft_mrif_3_2.cu', 'r') as f:
        content = f.read()
        content = re.sub(r'int mix_mrif_task_blk_num = \d+;', f'int mix_mrif_task_blk_num = {mix_mrif_task_blk_num};', content)

    
    # 将修改后的内容写回文件
    with open('fft_mrif_3_2.cu', 'w') as f:
        f.write(content)

    # 编译
    try:
        output = subprocess.check_output(compile_command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling fft_mrif_3_2_mix, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
        exit(1)
    if "Error" in output or "error" in output or "ERROR" in output:
        print(f"Error compiling fft_mrif_3_2_mix, Error: ---\n{output}\n---\n, Exit!", file=sys.stderr)
        exit(1)

def run():
    # 运行
    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error running {command}, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
        exit(1)
    # if "Error" in output or "error" in output or "ERROR" in output:
    #     print(f"Error running {command}, Error: ---\n{output}\n---\n, Exit!", file=sys.stderr)
    #     exit(1)
    # print(output)
    # 提取运行时间的数字部分
    pattern = re.compile(r'\[(\w+)\] improvement: (-?[1-9]\d*\.\d+)%')
    runtime_matches = pattern.findall(output)
    if runtime_matches:
        for match in runtime_matches:
            _, improvement = match
            global best_improve
            best_improve = max(best_improve, float(improvement))
            return float(improvement)
    else:
        print("No match found!", file=sys.stderr)
        exit(1)
            

if __name__ == "__main__":
    for i in [0, 256, 512, 1024]:
        compile(i)
        cur_improve = run()
        print(f"i: {i}  best:{best_improve}%  current:{cur_improve}%")
        # input("Press Enter to continue...")
