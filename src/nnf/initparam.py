import re

def extract_parameter_info(code: str):
    input_params = []
    output_params = []
    # fill_values = {}

    # Extract input parameters and sizes
    input_matches = re.findall(r'input argument\s*(\w+)\*\s*(\w+),\s*\*(\w+);.*?cudaMallocHost.*?sizeof\(\w+\)\s*\*\s*(\d+)', code, re.DOTALL)
    for var_type, host_var, device_var, size in input_matches:
        input_params.append((var_type, device_var, int(size)))

    # Extract output parameters and sizes
    output_matches = re.findall(r'output arguments\s*(\w+)\*\s*(\w+),\s*\*(\w+);.*?cudaMallocHost.*?sizeof\(\w+\)\s*\*\s*(\d+)', code, re.DOTALL)
    for var_type, host_var, device_var, size in output_matches:
        output_params.append((var_type, device_var, int(size)))

    # Extract fill input values iterations
    # fill_matches = re.findall(r'fill input values\s*for\s*\(int\s*i\s*=\s*0;\s*i\s*<\s*(\d+);', code)
    # if fill_matches:
    #     fill_values = {input_params[0][0] + "_host": int(fill_matches[0])}

    return input_params, output_params

input_arg_block = """
    //input argument
    {input_type}* {input_name}_host, *{input_name};
    CUDA_SAFE_CALL(cudaMallocHost((void**)&{input_name}_host, sizeof({input_type})* {input_size}));
    CUDA_SAFE_CALL(cudaMalloc((void**)&{input_name}, sizeof({input_type}) * {input_size}));
    for (int i = 0; i < {input_size}; ++i) {input_name}_host[i] = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpy({input_name}, {input_name}_host, sizeof({input_type}) * {input_size}, cudaMemcpyHostToDevice));
    this->Input[{no}] = {input_name};
    this->InputHost[{no}] = {input_name}_host;
    this->InputSize[{no}] = {input_size};
"""

initParams_template = """
#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/{model_name}/{model_name}.h"
#include "./dnn/{model_name}/{model_name}_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void {class_model_name}::initParams() {{
    {model_name}_cuda_init();
    {input_code}
    //output arguments
    float* {output_name}_host, *{output_name};
    CUDA_SAFE_CALL(cudaMallocHost((void**)&{output_name}_host, sizeof(float) * {output_size}));

    this->Result = (void**)&{output_name};

    //fill input values
    """