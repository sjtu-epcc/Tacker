# Aker

Tensor-CUDA Core kernel fusion system for improving the GPU utilization

Paper link:
* https://doi.org/10.1109/HPCA53966.2022.00064
* https://doi.org/10.1109/TC.2024.3477995
## kernel fusion

The script `fusion/code/gen_run.py` is used to generate fused kernel code from the candidate kernel. Kernel code will be generated in the `fusion/mix_kernel` directory, test code will be stored at `fusion/run` directory.

```bash
cd fusion/code
python gen_run.py

// run the test code
cd ../run
python run.py
```

You can find all benchmark original code in  `src/raw_kernel/`.

## DNN model

We use nnfusion to generate dnn model code. Refer to [microsoft/nnfusion](https://github.com/microsoft/nnfusion/tree/v0.4).

*(notes: we use nnfusion v0.4 built from source, so that transformer-based models can be supported.)*

After building nnfusion, you can use the following command to generate code for the dnn model([tensorflow example](https://github.com/microsoft/nnfusion/tree/v0.4/models/tensorflow)).

```bash
cd src/nnf/nnf_tf_freezer
python3 example.py --model_name=vgg16 --frozen_graph=vgg16.pb

nnfusion vgg16.pb
```

nnfusion can also gen code from onnx model.

```bash
nnfusion /path/to/pt-bert-base-cased.onnx -f onnx -p 'batch:3;sequence:512'
```

Finally, rewrite the kernel code to the `src/dnn` directory. We warp the generated code into proprietary classes to make it easier to use in runtime.

```bash
cd src/nnf
python gen_code.py
```
## runtime

For quick start, we provided dockerfile. To build the docker image, run

```bash
bash docker/build.sh
// then
bash docker/run.sh
```

in docker,

```bash
cd /workspace/aker
mkdir build && cd build
cmake .. && make -j
```

run the main code

```bash
./aker -s aker -m vgg16
```

## Citation

If you use Tacker / Aker for your research, please cite our papers:

```txt
@INPROCEEDINGS{9773253,
  author={Zhao, Han and Cui, Weihao and Chen, Quan and Zhang, Youtao and Lu, Yanchao and Li, Chao and Leng, Jingwen and Guo, Minyi},
  booktitle={2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA)}, 
  title={Tacker: Tensor-CUDA Core Kernel Fusion for Improving the GPU Utilization while Ensuring QoS}, 
  year={2022},
  volume={},
  number={},
  pages={800-813},
  keywords={Tensors;Runtime;Graphics processing units;Quality of service;Machine learning;Parallel processing;Throughput;Tensor Core;GPU Utilization;QoS},
  doi={10.1109/HPCA53966.2022.00064}}
```

```txt
@ARTICLE{10713257,
  author={Zhao, Han and Deng, Junxiao and Cui, Weihao and Chen, Quan and Zhang, Youtao and Zeng, Deze and Guo, Minyi},
  journal={IEEE Transactions on Computers}, 
  title={Adaptive Kernel Fusion for Improving the GPU Utilization while Ensuring QoS}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Kernel;Graphics processing units;Quality of service;Tensors;Processor scheduling;Throughput;Resource management;Computers;Instruction sets;Benchmark testing;Kernel fusion;QoS;GPU scheduling},
  doi={10.1109/TC.2024.3477995}}
```