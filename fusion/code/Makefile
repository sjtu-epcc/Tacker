cp:
	nvcc -o cp_out -arch=sm_75 -lm -lpthread --ptxas-options=-v my_cp.cu
cutcp:
	nvcc -o cutcp_out -arch=sm_75 -lcublas -lcurand -lm --ptxas-options=-v -Xptxas -dlcm=ca file_t/readatom.o my_cutcp.cu
fft:
	nvcc -o fft_out -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_fft.cu
lbm:
	nvcc -o lbm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_lbm.cu
mrif:
	nvcc -o mrif_out -arch=sm_75 -lm -lpthread --ptxas-options=-v my_mrif.cu
mriq:
	nvcc -o mriq_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_mriq.cu
sgemm:
	nvcc -o sgemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_sgemm.cu
stencil:
	nvcc -o stencil_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_stencil.cu

all: cp cutcp fft lbm mrif mriq sgemm tpacf
	@echo "All finished"
clean:
	rm *_out