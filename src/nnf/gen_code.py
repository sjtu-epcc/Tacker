
import re

def get_func(lines, function_name):
    start_index = None
    brace_count = 0
    function_content = []

    for i, line in enumerate(lines):
        if function_name + '(' in line:
            start_index = i
            brace_count += line.count('{') - line.count('}')
            function_content.append(line)
            break

    if start_index is None:
        print(f'Function {function_name} not found')
        exit(1)

    for line in lines[start_index + 1:]:
        brace_count += line.count('{')
        brace_count -= line.count('}')
        function_content.append(line)
        if brace_count == 0:
            break

    return '\n'.join(function_content)

def remove_func(lines, function_name):
    # 删除函数后返回新的代码
    in_func = False
    brace_count = 0
    new_content = []
    
    for i, line in enumerate(lines):
        if function_name + '(' in line:
            brace_count += line.count('{') - line.count('}')
            in_func = True
            continue
        if in_func:
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0:
                in_func = False
                continue
        else:
            new_content.append(line)
        
    return '\n'.join(new_content)
        

def extrace_args(raw_args):
    # 正则表达式用于匹配参数类型和参数名
    args_pattern = re.compile(r'(\w[\w\s\*]*\s\*?&?)\s*(\w+)')
    args = re.findall(args_pattern, raw_args)
    return args

def extract_func(func_name, raw_code):
    # 从代码中提取函数
    func_pattern = re.compile(rf'{func_name}\((.*?)\)\s*{{(.+?)}}', re.DOTALL)
    func = re.findall(func_pattern, raw_code)[0]
    args, body = func
    args = extrace_args(args)
    # 从代码中提取func_name对应的整个函数，从签名到最后一行
    raw_kernel = get_func(raw_code.split('\n'), func_name)
    return args, raw_kernel


class KernelClassCppGenerator:
    def __init__(self, model_name, code):
        self.raw_code = code
        self.model_name = model_name
        # self.cuda_init = self.gen_cuda_init()
        self.function_calls, self.entry_params_list = self.extract_kernel_entry_functions()

    def get_class_name(self):
        dic = {
            "bert": "Bert",
            "inception3": "Inception3",
            "lstm": "LSTM",
            "resnet50": "Resnet50",
            "vgg11": "VGG11",
            "vgg16": "VGG16",
            "vit": "ViT"
        }
        return dic[self.model_name]

    def process_params(self, params):
        # 手动拆分参数，处理嵌套的括号
        split_params = []
        start = 0
        bracket_level = 0

        for i, char in enumerate(params):
            if char == '(':
                bracket_level += 1
            elif char == ')':
                bracket_level -= 1
            elif char == ',' and bracket_level == 0:
                split_params.append(params[start:i].strip())
                start = i + 1

        # 添加最后一个参数
        if start < len(params):
            split_params.append(params[start:].strip())

        return split_params

    # def gen_cuda_init(self):
    #     cuda_init_pattern = re.compile(r'extern "C" void cuda_init\(.*?\)\s*{(.+?)}', re.DOTALL)
    #     cuda_init = re.findall(cuda_init_pattern, self.raw_code)[0]
    #     cuda_init = str(cuda_init.strip())
    #     cuda_init = cuda_init.replace('CUDA_SAFE_CALL(cudaDeviceReset());\n', '')
    #     return cuda_init

    def extract_function_calls(self, body)->list[tuple[str, list[str]]]:
        function_call_pattern = re.compile(r'\n(\w+)\s*\(([^;]*?)\)\s*;')
        function_calls = re.findall(function_call_pattern, body)
        processed_calls = []

        for func_name, params in function_calls:
            processed_params = self.process_params(params)
            processed_calls.append((func_name, processed_params))

        return processed_calls

    def extract_kernel_entry_functions(self)->tuple[list[tuple[str, list[str]]], list[str]]:
        kernel_entry_pattern = re.compile(r'extern "C" int kernel_entry\((.*?)\)\s*{(.+?)}', re.DOTALL)
        kernel_entries = re.findall(kernel_entry_pattern, self.raw_code)

        entry_params, body = kernel_entries[0]

        function_calls = self.extract_function_calls(body)
        entry_params_list = extrace_args(entry_params)
        # print("function_calls:", function_calls)
        
        return function_calls, entry_params_list

    def generate_class_code(self, call_function_name):
        class_name = f'{self.model_name + "_" + call_function_name}Kernel'

        sig_args, raw_kernel = extract_func(call_function_name, self.raw_code)

        mix_able = 0

        get_args_code = """
std::vector<int> getArgs() override {
    return std::vector<int>();
}"""

        if ("cublasSgemm(" in raw_kernel):
            mix_able = 1
            from mysgemm import extract_mnk
            # print(raw_kernel)
            m, n, k = extract_mnk(raw_kernel)
            get_args_code = f"""
std::vector<int> getArgs() override {{
    std::vector<int> ret(3);
    ret[0] = {m};
    ret[1] = {n};
    ret[2] = {k};
    return ret;
}}"""

        elif ("cudnnConvolutionForward" in raw_kernel):
            mix_able = 2
            from mycudnn import extract_dimensions
            input_dims, kernel_dims, conv_dims = extract_dimensions(raw_kernel)
            get_args_code = f"""
std::vector<int> getArgs() override {{
    return std::vector<int>({{{input_dims[0]}, {input_dims[1]}, {input_dims[2]}, {input_dims[3]}, {kernel_dims[0]}, {kernel_dims[1]}, {kernel_dims[2]}, {kernel_dims[3]}, {conv_dims[0]}, {conv_dims[1]}, {conv_dims[2]}, {conv_dims[3]}, {conv_dims[4]}, {conv_dims[5]}}});
}}""" # input 0-4, kernel 5-8, conv 9-14
        # 双指针匹配参数，将签名中的参数名替换为调用时的参数名。如果调用时参数以dim开头或者为字面量，则该调用参数跳过
        construct_args = []
        i = 0
        while i < len(sig_args):
            construct_args.append((sig_args[i][0], sig_args[i][1]))
            i += 1

        return f'''
class {class_name} : public Kernel {{
public:
    {class_name}({", ".join([f'{arg_type} {arg_name}' for arg_type, arg_name in (construct_args + self.entry_params_list)])}) {{
        {", ".join([f'this->{arg_name} = {arg_name}' for _, arg_name in (construct_args + self.entry_params_list)])};
        this->kernelName = "{self.model_name + "_" + call_function_name}";
        this->Id = 0;
        this->mixable = {mix_able};
    }}

    void initParams() override {{
        // empty implementation
    }}

    void loadKernel() override {{
        // Empty implementation
    }}

        {"; ".join([f'{arg_type.replace("const ", "").replace("&", "")} {arg_name}' for arg_type, arg_name in construct_args])};
    {"; ".join([f'{arg_type.replace("const ", "").replace("&", "")} {arg_name}' for arg_type, arg_name in self.entry_params_list])};
private:

    {get_args_code}

    {raw_kernel.replace('extern "C" ', '')
        .replace('extern', '')
        .replace("(cudnnConvolutionForward(", "(mycudnnConvolutionForward(")
        .replace("(cublasSgemm(", "(mycublasSgemm(")}

    void executeImpl(cudaStream_t stream) {{
        this->{call_function_name}({", ".join([f'{arg_name}' for _, arg_name in construct_args])});
    }}
}};
'''

    def generate_all_classes(self):
        class_list = {}
        counts = 0
        for name, params in self.function_calls:
            class_list[name] = self.generate_class_code(name)
            counts += 1
            # print(f"generate {counts} classes")

        class_codes = []
        for name, code in class_list.items():
            class_codes.append(code)
        return '\n'.join(class_codes)

    def gen_vector_code(self):
        code = f'''void {self.get_class_name()}::gen_vector({", ".join([f'{arg_type.replace("const ", "")} {arg_name}' for arg_type, arg_name in self.entry_params_list])}) {{\n'''
        for name, params in self.function_calls:
            arg_name_list = []
            for param in params:
                arg_name_list.append(param)
            for _, arg_name in self.entry_params_list:
                arg_name_list.append(arg_name)
            for i in range(len(arg_name_list)):
                if not (str(arg_name_list[i]).isdigit() or str(arg_name_list[i]).find("dim") != -1):
                    arg_name_list[i] = f'std::move({arg_name_list[i]})'
            new_line = f'    kernels.emplace_back(new {self.model_name + "_" + name}Kernel({", ".join([f"{arg_name}" for arg_name in arg_name_list])}));\n'
            if "dim3" in new_line:
                pattern = r'dim3(\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)), dim3(\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)), (\d+), (\d+),'
                # 替换第4个参数为nullptr
                new_line = re.sub(pattern, r'dim3\1, dim3\2, \3, nullptr,', new_line)
            code += new_line
        code += '}\n'
        return code




bin_dir = './{model_name}/cuda_codegen/Constant'
code_file = './{model_name}/cuda_codegen/nnfusion_rt.cu'
code_dir = './{model_name}/cuda_codegen'
gen_dir = '../dnn'
# dnn_list = ["inception3", "resnet50", "vgg11", "vgg16", "bert", "vit"]
dnn_list = ["vit"]
dnn_class_dict = {
    "inception3": "Inception3",
    "resnet50": "Resnet50",
    "vgg11": "VGG11",
    "vgg16": "VGG16",
    "bert": "Bert",
    "vit": "ViT"
}
result = {}

if __name__ == '__main__':
    for model_name in dnn_list:
        print(f"===================={model_name}====================")
        code_path = code_file.format(model_name=model_name)
        func_list = []
        with open(code_path, 'r') as f:
            nnfusion_code = f.read()
            generator = KernelClassCppGenerator(model_name, nnfusion_code)
            func_list = generator.function_calls
            with open(f'{gen_dir}/{model_name}/{model_name}_kernel_class.tmp', 'w') as f:
                f.write("""#pragma once
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#include <cuda.h>
#include <cuda_runtime.h>
""")               
                # from common import common_header
                # from mycudnn import mycudnnCovolutionForwardDef
                # from mysgemm import mycublasSgemmDef
                down_dict = {}
                begin_mem = False
                begin_persist = False
                end_mem = False
                mem_list = []
                constant_list = []
                for line in nnfusion_code.split('\n'):
                    if "char* group_0_CUDA_GPU0_allocator_memory_pool" in line:
                        begin_mem = True
                    if "memory_pool" in line and "char*" in line and "persist" in line:
                        begin_persist = True
                    if begin_persist and "Node name" in line:
                        break
                    if begin_mem and not begin_persist:
                        pattern = r"(\w+\*)\s*(\w+);"
                        matches = re.findall(pattern, line)
                        for match in matches:
                            mem_list.append((match[0], match[1]))
                    if begin_persist and not end_mem:
                        pattern = r"(\w+\*)\s*(\w+);"
                        matches = re.findall(pattern, line)
                        for match in matches:
                            constant_list.append((match[0], match[1]))
                # print("mem_list:", mem_list)
                # print("constant_list:", constant_list)
                # input("check")
                for name, _ in func_list:
                    if "_Call" in name and name not in down_dict:
                        nnfusion_code = nnfusion_code.replace("void " + name, "void " + model_name + "_" + name)
                        # nnfusion_code = nnfusion_code.replace(f'Constant_float_cuda_Constant_', f'{model_name}_Constant_float_cuda_Constant_')
                        down_dict[name] = True
                f.write(remove_func(nnfusion_code.split('\n'), 'kernel_entry')
                        .replace('CUDA_SAFE_CALL(cudaDeviceReset());\n', '')
                        .replace("CUDA_SAFE_CALL(cudaSetDevice(0));", '')
                        .replace('#include "nnfusion_rt.h"', '')
                        .replace('cuda_free', f'{model_name}_cuda_free')
                        .replace('"./Constant', f'"{gen_dir}/{model_name}/Constant'))
                f.write('#include "./include/dnn.h"\n')
                f.write(generator.generate_all_classes())
                f.write(generator.gen_vector_code())
        with open(f'{gen_dir}/{model_name}/{model_name}_kernel_class.tmp', 'r') as f:
            raw_code = f.read()
        with open(f'{gen_dir}/{model_name}/{model_name}_kernel_class.h', 'w') as f:
            down_dict = {}
            for name, _ in func_list:
                if "_Call" in name and name not in down_dict:
                    raw_code = raw_code.replace("__global__  void ", "__global__ void ")
                    raw_code = raw_code.replace("__global__ void " + name.replace("_Call", ""), "__global__ void " + model_name + "_" + name.replace("_Call", ""))
                    # print(f"{name.replace('_Call', '')}<<< --> {model_name + '_' + name.replace('_Call', '')}<<<")
                    raw_code = raw_code.replace(name.replace("_Call", "") + "<<<", model_name + "_" + name.replace("_Call", "") + "<<<")
                    # print(name)
                    down_dict[name] = True
            for _, name in mem_list:
                raw_code = raw_code.replace(name, model_name + "_" + name)
            for _, name in constant_list:
                raw_code = raw_code.replace(name, model_name + "_" + name)
            for _, name in constant_list:
                raw_code = raw_code.replace(model_name + "_" + name + ".bin", name + ".bin")
                # print(f"{model_name + '_' + name}.bin --> {name}.bin")
            # input("check")
            raw_code = raw_code.replace("Constant_float_cuda_Constant_", f'{model_name}_Constant_float_cuda_Constant_')
            raw_code = raw_code.replace("cuda_init", f'{model_name}_cuda_init')
            raw_code = raw_code.replace("cublas_handle_0", f'{model_name}_cublas_handle_0')
            # cudnn_handle_0
            raw_code = raw_code.replace("cudnn_handle_0", f'{model_name}_cudnn_handle_0')
            # num_SMs
            raw_code = raw_code.replace("num_SMs", f'{model_name}_num_SMs')
            raw_code = raw_code.replace("static bool selected_algo = false;", f'static bool selected_algo = true;')
            f.write(raw_code)
    # 将Constant文件夹拷贝到dnn文件夹下
    import shutil
    for model_name in dnn_list:
        # 删除{gen_dir}/{model_name}/Constant
        shutil.rmtree(f'{gen_dir}/{model_name}/Constant', ignore_errors=True)
        shutil.copytree(f'{bin_dir.format(model_name=model_name)}', f'{gen_dir}/{model_name}/Constant')

        # 更新{gen_dir}/{model_name}.cu
        # step1: 读取main_test.cpp
        with open(f'{code_dir.format(model_name=model_name)}/main_test.cpp', 'r') as f:
            main_test = f.read()

        with open(f'{gen_dir}/{model_name}/{model_name}.cu', 'w') as f:
            from initparam import initParams_template, input_arg_block, extract_parameter_info
            input_params, output_params = extract_parameter_info(main_test)
            input_code = ""
            if model_name == "bert":
                assert len(input_params) == 3
            else:
                assert len(output_params) == 1
            for i in range(len(input_params)):
                input_code += input_arg_block.format(input_type=input_params[i][0],
                                                    input_name=input_params[i][1], 
                                                    input_size=input_params[i][2], 
                                                    no=i)
            # print(f"input_code: {input_code}")
            content = initParams_template.format(model_name=model_name, 
                                                 class_model_name=dnn_class_dict[model_name], 
                                                 input_code=input_code, 
                                                 output_name=output_params[0][1],
                                                 output_size=output_params[0][2])
            gen_vector = "this->gen_vector("
            for i in range(len(input_params)):
                gen_vector += input_params[i][1] + ", "
            gen_vector += f"({output_params[0][0]}**)" + "Result);\n}"
            content += gen_vector
            f.write(content)