#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>


class Kernel {
public:
    template <typename... Args>
    void execute(Args&&... args) {
        executeImpl(std::forward<Args>(args)...);
    }
    virtual void initParams() = 0;
    virtual void loadKernel() = 0;

    virtual ~Kernel() = default; // 声明虚析构函数，以确保子类的析构函数被调用

    int getKernelId() { return Id; }
    std::string& getKernelName() { return kernelName; }

    int Id;
    std::string kernelName;
    std::string moduleName;
    CUfunction function;
    unsigned int smem;
    std::vector<void*> kernelParams;
    std::vector<void*> cudaFreeList;
    dim3 launchGridDim;
    dim3 launchBlockDim;

    void* kernelFunc;
protected:
    virtual void executeImpl() = 0; // 纯虚函数，由子类实现
};

class Add_float_float_float_cuda_Add_504_CallKernel : public Kernel {
public:
    Add_float_float_float_cuda_Add_504_CallKernel(dim3 grids, dim3 blocks, unsigned  mem, cudaStream_t  stream, float*  input0, float*  input1, float*  output0, float*  Parameter_270_0, float**  Result_505_0) {
        this->grids = grids, this->blocks = blocks, this->mem = mem, this->stream = stream, this->input0 = input0, this->input1 = input1, this->output0 = output0, this->Parameter_270_0 = Parameter_270_0, this->Result_505_0 = Result_505_0;
    }

    void initParams() override {
        // empty implementation
    }

    void loadKernel() override {
        // Empty implementation
    }
private:
    dim3  grids; dim3  blocks; unsigned  mem; cudaStream_t  stream; float*  input0; float*  input1; float*  output0;
    float*  Parameter_270_0; float**  Result_505_0;

    void executeImpl() {
        // Empty implementation
    }

};



int main() {
    auto ins_vec = std::queue<std::unique_ptr<Add_float_float_float_cuda_Add_504_CallKernel>>{};

    auto float_ptr1 = (float*)malloc(sizeof(float));
    auto float_ptr2 = (float*)malloc(sizeof(float));
    // 使用std::make_unique创建MyClass的实例
    ins_vec.push(std::make_unique<Add_float_float_float_cuda_Add_504_CallKernel>(dim3(1,2,3), dim3(4,5,6), 0, nullptr, std::move(float_ptr1), std::move(float_ptr2), nullptr, nullptr, nullptr));

    // 使用myClassInstance...
    
    return 0;
}
