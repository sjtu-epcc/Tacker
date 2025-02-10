import re

def extract_dimensions(code_str):
    # 正则表达式匹配输入、核心和卷积的维度
    input_pattern = r'cudnnSetTensor4dDescriptor\(([^,]+), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (\d+), (\d+), (\d+), (\d+)\)'
    kernel_pattern = r'cudnnSetFilter4dDescriptor\(([^,]+), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, (\d+), (\d+), (\d+), (\d+)\)'
    conv_pattern = r'cudnnSetConvolution2dDescriptor\(([^,]+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT\)'

    inputs = re.findall(input_pattern, code_str)
    kernels = re.findall(kernel_pattern, code_str)
    convs = re.findall(conv_pattern, code_str)

    # 提取的结果可能包含多个匹配，这里只获取第一个匹配
    input_dims = [int(dim) for dim in inputs[0][1:]] if inputs else None
    kernel_dims = [int(dim) for dim in kernels[0][1:]] if kernels else None
    conv_dims = [int(dim) for dim in convs[0][1:]] if convs else None  

    return input_dims, kernel_dims, conv_dims