import re
import pandas as pd

def extract_data(file_path):
    data = []
    pattern = r'--- (\d+) ---\s+load_ratio:\s+(\d+\.\d+)\s+mix_duration:\s+(\d+\.\d+)\s+.*cp_blk_num:\s+(\d+),\s+sgemm_blk_num:\s+(\d+)'

    with open(file_path, 'r') as file:
        file_content = file.read()
        matches = re.findall(pattern, file_content, re.MULTILINE)

        for match in matches:
            data.append({
                'number': int(match[0]),
                'load_ratio': float(match[1]),
                'mix_duration': float(match[2]),
                'cp_blk_num': int(match[3]),
                'sgemm_blk_num': int(match[4])
            })

    return data

def write_to_excel(data, output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

# Replace 'your_file.txt' with the path to your text file
file_path = 'cp_sgemm_1_1.log'
data = extract_data(file_path)

# Replace 'output.xlsx' with your desired output Excel file name
output_file = 'cp_sgemm_load_ratio_data.xlsx'
write_to_excel(data, output_file)
