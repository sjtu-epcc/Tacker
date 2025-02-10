import re
import pandas as pd

def extract_data(file_path):
    data = []
    # pattern = r'--- (\d+) ---\s+load_ratio:\s+(\d+\.\d+)\s+mix_duration:\s+(\d+\.\d+)\s+.*cp_blk_num:\s+(\d+)/\d+,\s+sgemm_blk_num:\s+(\d+)/\d+'
    pattern = r'load_ratio:\s+(\d+\.\d+)\s+mix_duration:\s+(\d+\.\d+)\s+.*cp gptb time:\s+(\d+\.\d+)\s+.*sgemm gptb time:\s+(\d+\.\d+)\s+.*cp_blk_num:\s+(\d+)/\d+,\s+sgemm_blk_num:\s+(\d+)/\d+'
    with open(file_path, 'r') as file:
        file_content = file.read()
        matches = re.findall(pattern, file_content, re.MULTILINE)

        for match in matches:
            if int(match[4]) >= 320 * 512 or int(match[5]) >= 32 * 258:
                continue
            else:
                data.append({
                    'load_ratio': float(match[0]),
                    'mix_duration': float(match[1]),
                    'cp_gptb_time': float(match[2]),
                    'sgemm_gptb_time': float(match[3]),
                    'cp_blk_num': int(match[4]),
                    'sgemm_blk_num': int(match[5])
                })

    return data

def write_to_excel(data, output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

# Replace 'your_file.txt' with the path to your text file
file_path = 'cp_sgemm_1_1_scale_rate.log'
data = extract_data(file_path)

# Replace 'output.xlsx' with your desired output Excel file name
output_file = 'cp_sgemm_load_ratio_data2_scale_rate.xlsx'
write_to_excel(data, output_file)
