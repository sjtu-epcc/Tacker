import re
def extract_mnk(code_str):
    # cublasSgemm(cublas_handle, XXX, XXX, 1001, 1, 2048
    cublas_invoke_pattern = r'cublasSgemm\(cublas_handle, \S+, \S+, (\d+), (\d+), (\d+)'

    sgemm_args = re.findall(cublas_invoke_pattern, code_str)    

    # 提取的结果可能包含多个匹配，这里只获取第一个匹配
    sgemm_dims = [int(dim) for dim in sgemm_args[0][0:]] if sgemm_args else None

    return sgemm_dims[0], sgemm_dims[1], sgemm_dims[2]