import re
import subprocess
import sys

compile_command = "make fft_sgemm_mix -j $(nproc)"
command = "./fft_sgemm_1_4_mix"

def compile(mix_fft_task_blk_num):
    with open('fft_sgemm_1_4.cu', 'r') as f:
        content = f.read()
        content = re.sub(r'int mix_fft_task_blk_num = \d+;', f'int mix_fft_task_blk_num = {mix_fft_task_blk_num};', content)

    
    # 将修改后的内容写回文件
    with open('fft_sgemm_1_4.cu', 'w') as f:
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
    # if extract_data(output)['load_ratio'] > 1.0:
    #     return 
    data_dic_list = []
    for i in range(10):
        try:
            output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error running {command}, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
            exit(1)
        data_dic_list.append(extract_data(output))
    
    # 对mix_duration进行排序，取中间10次的平均值
    data_dic_list.sort(key=lambda x: x['mix_duration'])
    data_dic_list = data_dic_list[2:7]
    load_ratio = sum([x['load_ratio'] for x in data_dic_list]) / len(data_dic_list)
    mix_duration = sum([x['mix_duration'] for x in data_dic_list]) / len(data_dic_list)
    sgemm_gptb_time = sum([x['sgemm_gptb_time'] for x in data_dic_list]) / len(data_dic_list)
    fft_gptb_time = sum([x['fft_gptb_time'] for x in data_dic_list]) / len(data_dic_list)
    sgemm_blk_num = sum([x['sgemm_blk_num'] for x in data_dic_list]) / len(data_dic_list)
    fft_blk_num = sum([x['fft_blk_num'] for x in data_dic_list]) / len(data_dic_list)
    return {
        'load_ratio': load_ratio,
        'mix_duration': mix_duration,
        'sgemm_gptb_time': sgemm_gptb_time,
        'fft_gptb_time': fft_gptb_time,
        'sgemm_blk_num': sgemm_blk_num,
        'fft_blk_num': fft_blk_num
    }

import re
import pandas as pd

data = []

def extract_data(data_content):
    data_ = []
    pattern = r'load_ratio:\s+(\d+\.\d+)\s+mix_duration:\s+(\d+\.\d+)\s+.*sgemm gptb time:\s+(\d+\.\d+), fft gptb time:\s+(\d+\.\d+), sgemm_blk_num:\s+(\d+),\s+fft_blk_num:\s+(\d+)'

    matches = re.findall(pattern, data_content, re.MULTILINE)

    for match in matches:
        data_.append({
            'load_ratio': float(match[0]),
            'mix_duration': float(match[1]),
            'sgemm_gptb_time': float(match[2]),
            'fft_gptb_time': float(match[3]),
            'sgemm_blk_num': int(match[4]),
            'fft_blk_num': int(match[5])
        })
    return data_[0]

def write_to_excel(output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    max_fft_blks = 10240 * 20
    for i in range(142 * 50, max_fft_blks + 1):
        if i % (142 * 50) == 0:
            print(f"--- {i} ---")
            compile(i)
            output = run()
            if output['load_ratio'] > 1.2:
                break
            print(output)
            data.append(output)
        
        # input("Press Enter to continue...")

    # Replace 'output.xlsx' with your desired output Excel file name
    output_file = '[Aker]fig9-fft-sgemm.xlsx'
    write_to_excel(output_file)

