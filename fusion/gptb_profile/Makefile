cp:
	nvcc -o cp_out -I../ -arch=sm_75 -lm -lpthread --ptxas-options=-v cp_gptb_profile.cu
cutcp:
	nvcc -o cutcp_out -I../ -arch=sm_75 -lcublas -lcurand -lm --ptxas-options=-v -Xptxas -dlcm=ca ../file_t/readatom.o cutcp_gptb_profile.cu
fft:
	nvcc -o fft_out -I../ -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v fft_gptb_profile.cu
lbm:
	nvcc -o lbm_out -I../ -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca lbm_gptb_profile.cu
mrif:
	nvcc -o mrif_out -I../ -arch=sm_75 -lm -lpthread --ptxas-options=-v mrif_gptb_profile.cu
mriq:
	nvcc -o mriq_out -I../ -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mriq_gptb_profile.cu
sgemm:
	nvcc -o sgemm_out -I../ -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca sgemm_gptb_profile.cu
stencil:
	nvcc -o stencil_out -I../ -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca stencil_gptb_profile.cu
all: cp cutcp fft lbm mrif mriq sgemm stencil
	@echo "All finished"
clean:
	rm *_out