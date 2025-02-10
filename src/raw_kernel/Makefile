hot3d:
	nvcc -o hot3d_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ori_hot3d.cu
lava:
	nvcc -o lava_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ori_lava.cu
path:
	nvcc -o path_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ori_path.cu
nn:
	nvcc -o nn_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ori_nn.cu

all: hot3d lava path nn
clean:
	rm *_out