# 首先运行mix_kernels/code/gptb_profile.py生成数据
import re

import pandas as pd

data = []

writer = pd.ExcelWriter(f"gptb_profile.xlsx")

def extract_data(kernel, file_path):
    data.clear()
    with open(file_path, 'r') as f:
        text = f.read()

    # 定义三个正则表达式模式分别用于提取ORI、PTB和GPTB的数据
    ori_pattern = fr'\[ORI\]\s+{kernel}_grid[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+\*\s+(\d+)\s+{kernel}_block[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+\*\s+(\d+)\s+\[ORI\]\s+{kernel}\s+took\s+([\d.]+)\s+ms'
    ptb_pattern = fr'\[PTB\]\s+{kernel}_grid[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+\*\s+(\d+)\s+{kernel}_block[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+\*\s+(\d+)\s+\[PTB\]\s+{kernel}\s+took\s+([\d.]+)\s+ms'
    gptb_pattern = fr'\[GPTB\]\s+{kernel}\s+took\s+([\d.]+)\s+ms\n\[GPTB\]\s+{kernel}\s+blks:\s+(\d+)'

    if kernel == "sgemm":
        ori_pattern = fr'\[ORI\]\s+{kernel}_grid[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+{kernel}_block[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+\[ORI\]\s+{kernel}\s+took\s+([\d.]+)\s+ms'
        ptb_pattern = fr'\[PTB\]\s+{kernel}_grid[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+{kernel}_block[2]*\s+--\s+(\d+)\s+\*\s+(\d+)\s+\[PTB\]\s+{kernel}\s+took\s+([\d.]+)\s+ms'

    # 使用正则表达式模式匹配文本并提取数据
    ori_matches = re.findall(ori_pattern, text)
    ptb_matches = re.findall(ptb_pattern, text)
    gptb_matches = re.findall(gptb_pattern, text)

    # 创建三个向量分别用于存储ORI、PTB和GPTB的数据
    ori_data = []
    ptb_data = []
    gptb_data = []

    if kernel == "sgemm":
        for match in ori_matches:
            grid_x = int(match[0])
            grid_y = int(match[1])
            block_x = int(match[2])
            block_y = int(match[3])
            time = float(match[4])
            ori_data.append([grid_x, grid_y, block_x, block_y, time])

        for match in ptb_matches:
            grid_x = int(match[0])
            grid_y = int(match[1])
            block_x = int(match[2])
            block_y = int(match[3])
            time = float(match[4])
            ptb_data.append([grid_x, grid_y, block_x, block_y, time])

        for match in gptb_matches:
            time = float(match[0])
            blks = int(match[1])
            gptb_data.append([time, blks])
        if len(ori_data) != len(ptb_data) or len(ptb_data) != len(gptb_data):
            print(f"Error: {kernel} data length not match!")
            print(f"ORI: {len(ori_data)}")
            print(f"PTB: {len(ptb_data)}")
            print(f"GPTB: {len(gptb_data)}")
            return
        for i in range(len(ori_data)):
            line_data = []
            for ii in range(len(ori_data[i])):
                line_data.append(ori_data[i][ii])
                if ii == 1:
                    line_data.append(1)
                if ii == 3:
                    line_data.append(1)
            for ii in range(len(ptb_data[i])):
                line_data.append(ptb_data[i][ii])
                if ii == 1:
                    line_data.append(1)
                if ii == 3:
                    line_data.append(1)
            for ii in range(len(gptb_data[i])):
                line_data.append(gptb_data[i][ii])
            data.append(line_data)
            if i == 0:
                print(line_data)
                input()
        # 创建DataFrame
        df = pd.DataFrame(data)

        df.to_excel(writer, index=False, header=["ORI Grid X", "ORI Grid Y", "ORI Grid Z","ORI Block X", "ORI Block Y", "ORI Block Z", "ORI Time (ms)", "PTB Grid X", "PTB Grid Y", "PTB Grid Z", "PTB Block X", "PTB Block Y", "PTB Block Z", "PTB Time (ms)", "GPTB Time (ms)", "GPTB Blks"], sheet_name=f"{kernel}")
        return
    elif kernel == "stencil":
        print("stencil")
        for match in ori_matches:
            grid_x = int(match[0])
            grid_y = int(match[1])
            grid_z = int(match[2])
            block_x = int(match[3])
            block_y = int(match[4])
            block_z = int(match[5])
            time = float(match[6])
            ori_data.append([grid_x, grid_y, grid_z, block_x, block_y, block_z, time])

        # 提取GPTB的匹配数据并存储
        for match in gptb_matches:
            time = float(match[0])
            blks = int(match[1])
            gptb_data.append([time, blks])

        if len(ori_data) != len(gptb_data):
            print(f"Error: {kernel} data length not match!")
            print(f"ORI: {len(ori_data)}")
            print(f"GPTB: {len(gptb_data)}")
            return
        
        for i in range(len(ori_data)):
            line_data = []
            for ii in range(len(ori_data[i])):
                line_data.append(ori_data[i][ii])
            for ii in range(len(gptb_data[i])):
                line_data.append(gptb_data[i][ii])
            data.append(line_data)
            if i == 0:
                print(line_data)
                input()

        # 创建DataFrame
        df = pd.DataFrame(data)
        print(data[0])

        df.to_excel(writer, index=False, header=["ORI Grid X", "ORI Grid Y", "ORI Grid Z", "ORI Block X", "ORI Block Y", "ORI Block Z", "ORI Time (ms)", "GPTB Time (ms)", "GPTB Blks"], sheet_name=f"{kernel}")

    else:
        # 提取ORI的匹配数据并存储
        for match in ori_matches:
            grid_x = int(match[0])
            grid_y = int(match[1])
            grid_z = int(match[2])
            block_x = int(match[3])
            block_y = int(match[4])
            block_z = int(match[5])
            time = float(match[6])
            ori_data.append([grid_x, grid_y, grid_z, block_x, block_y, block_z, time])

        # 提取PTB的匹配数据并存储
        for match in ptb_matches:
            grid_x = int(match[0])
            grid_y = int(match[1])
            grid_z = int(match[2])
            block_x = int(match[3])
            block_y = int(match[4])
            block_z = int(match[5])
            time = float(match[6])
            ptb_data.append([grid_x, grid_y, grid_z, block_x, block_y, block_z, time])

        # 提取GPTB的匹配数据并存储
        for match in gptb_matches:
            time = float(match[0])
            blks = int(match[1])
            gptb_data.append([time, blks])

        if len(ori_data) != len(ptb_data) or len(ptb_data) != len(gptb_data):
            print(f"Error: {kernel} data length not match!")
            print(f"ORI: {len(ori_data)}")
            print(f"PTB: {len(ptb_data)}")
            print(f"GPTB: {len(gptb_data)}")
            return
        
        for i in range(len(ori_data)):
            line_data = []
            for ii in range(len(ori_data[i])):
                line_data.append(ori_data[i][ii])
            for ii in range(len(ptb_data[i])):
                line_data.append(ptb_data[i][ii])
            for ii in range(len(gptb_data[i])):
                line_data.append(gptb_data[i][ii])
            data.append(line_data)
            if i == 0:
                print(line_data)
                input()

        # 创建DataFrame
        df = pd.DataFrame(data)

        df.to_excel(writer, index=False, header=["ORI Grid X", "ORI Grid Y", "ORI Grid Z", "ORI Block X", "ORI Block Y", "ORI Block Z", "ORI Time (ms)", "PTB Grid X", "PTB Grid Y", "PTB Grid Z", "PTB Block X", "PTB Block Y", "PTB Block Z", "PTB Time (ms)", "GPTB Time (ms)", "GPTB Blks"], sheet_name=f"{kernel}")
    

kernel_list = ['cp', 'cutcp', 'fft', 'lbm', 'mrif', 'mriq', 'sgemm', 'stencil']

if __name__ == "__main__":
    for kernel in kernel_list:
        extract_data(kernel, f"./{kernel}_gptb_profile.log")

    writer.close()