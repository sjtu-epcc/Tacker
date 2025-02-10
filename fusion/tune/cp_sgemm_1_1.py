import re
import subprocess
import sys

compile_command = "make cp_sgemm_mix -j $(nproc)"
command = "./cp_sgemm_1_1_mix"

best_improve = -11111.0

def compile(mix_mrif_task_blk_num):
    # open cp_sgemm_1_1.cu, replace mix_mrif_task_blk_num
    with open('cp_sgemm_1_1.cu', 'r') as f:
        content = f.read()
        content = re.sub(r'int mix_cp_task_blk_num = \d+;', f'int mix_cp_task_blk_num = {mix_mrif_task_blk_num};', content)

    
    # 将修改后的内容写回文件
    with open('cp_sgemm_1_1.cu', 'w') as f:
        f.write(content)

    # 编译
    try:
        output = subprocess.check_output(compile_command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error compile, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
        exit(1)
    # if "Error" in output or "error" in output or "ERROR" in output:
    #     print(f"Error compile, Error: ---\n{output}\n---\n, Exit!", file=sys.stderr)
    #     exit(1)

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
    # extrace time
    # pattern = re.compile(r'\[(\w+)\] improvement: (-?[1-9]\d*\.\d+)%')
    # runtime_matches = pattern.findall(output)
    # if runtime_matches:
    #     for match in runtime_matches:
    #         _, improvement = match
    #         global best_improve
    #         best_improve = max(best_improve, float(improvement))
    #         return float(improvement)
    
    # load_ratio: %f , cp gptb time: %f , sgemm gptb time: %f, cp_blk_num: %d, sgemm_blk_num: %d
    # pattern = re.compile(r'load_ratio: (\d+\.\d+) , cp gptb time: (\d+\.\d+) , sgemm gptb time: (\d+\.\d+), cp_blk_num: (\d+), sgemm_blk_num: (\d+)')
    # runtime_matches = pattern.findall(output)
    # if runtime_matches:
    #     for match in runtime_matches:
    #         load_ratio, cp_gptb_time, sgemm_gptb_time, cp_blk_num, sgemm_blk_num = match
    #         print(f"load_ratio: {load_ratio}, cp gptb time: {cp_gptb_time}, sgemm gptb time: {sgemm_gptb_time}, cp_blk_num: {cp_blk_num}, sgemm_blk_num: {sgemm_blk_num}")
    #         # TODO
    # # ori sum time: %f, fuse_solo time: %f, improvement: %f%
    # pattern = re.compile(r'ori sum time: (\d+\.\d+), fuse_solo time: (\d+\.\d+), improvement: (-?[1-9]\d*\.\d+)%')
    # runtime_matches = pattern.findall(output)
    # if runtime_matches:
    #     for match in runtime_matches:
    #         ori_sum_time, fuse_solo_time, improvement = match
    #         global best_improve
    #         best_improve = max(best_improve, float(improvement))
    #         print(f"ori sum time: {ori_sum_time}, fuse_solo time: {fuse_solo_time}, improvement: {improvement}%")
            
    print(output)
    # else:
    #     print("No match found!", file=sys.stderr)
    #     exit(1)
            

if __name__ == "__main__":
    max_cp_blks = 32 * 512
    for i in range(1, max_cp_blks + 1):
        if i % 256 == 0:
            print(f"--- {i} ---")
            compile(i)
            run()
        
        # input("Press Enter to continue...")
