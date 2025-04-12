PRJHOME = $(shell pwd)
CONDA = $(HOME)/miniconda3
CUDAHOME = $(HOME)/nvidia/cuda-12.1
CUDAINC = $(CUDAHOME)/include
CUDALIB = $(CUDAHOME)/lib64
PRJINC = $(PRJHOME)/Include
PRJLIB = $(PRJHOME)/Lib
#------------------------------
CUSRCS = ./Util/GAddImages.cu \
	 ./Util/GCalcMoment2D.cu \
	 ./Util/GNormalize2D.cu \
	 ./Util/GThreshold2D.cu \
	 ./FindCTF/GCalcCTF1D.cu \
	 ./FindCTF/GCalcCTF2D.cu \
	 ./FindCTF/GCalcSpectrum.cu \
	 ./FindCTF/GRadialAvg.cu \
	 ./FindCTF/GBackground1D.cu \
	 ./FindCTF/GRmBackground2D.cu \
	 ./FindCTF/GRemoveMean.cu \
	 ./FindCTF/GRoundEdge.cu \
	 ./FindCTF/GSpectralCC2D.cu \
	 ./FindCTF/GCC1D.cu \
	 ./FindCTF/GCC2D.cu \
	 ./FindCTF/GLowpass2D.cu
CUCPPS = $(patsubst %.cu, %.cpp, $(CUSRCS))
#------------------------------------------
SRCS = ./CInput.cpp \
	./Util/CParseArgs.cpp \
	./Util/CRegSpline.cpp \
	./Util/CRegSpline2.cpp \
	./Util/CRegSpline3.cpp \
	./Util/CSaveTempMrc.cpp \
	./Util/CCudaHelper.cpp \
	./MrcUtil/CLoadImages.cpp \
	./MrcUtil/CAsyncSaveImages.cpp \
	./MrcUtil/CSaveImages.cpp \
	./MrcUtil/CAsyncSingleSave.cpp \
	./FindCTF/CFindCtfHelp.cpp \
	./FindCTF/CCTFTheory.cpp \
	./FindCTF/CGenAvgSpectrum.cpp \
	./FindCTF/CSpectrumImage.cpp \
	./FindCTF/CFindDefocus1D.cpp \
	./FindCTF/CFindDefocus2D.cpp \
	./FindCTF/CFindCtf1D.cpp \
	./FindCTF/CFindCtf2D.cpp \
	./FindCTF/CFindCtfBase.cpp \
	./CCtfPackage.cpp \
	./CInputFolder.cpp \
	./CSaveCtfResults.cpp \
	./CLoadAngFile.cpp \
	./CFindSeriesCtfs.cpp \
	./CProcessThread.cpp \
	./CProcessMain.cpp \
	./CCtfFindMain.cpp \
	$(CUCPPS)
OBJS = $(patsubst %.cpp, %.o, $(SRCS))
#-------------------------------------
CC = g++
CFLAG = -c -g -pthread -m64
NVCC = $(CUDAHOME)/bin/nvcc -std=c++11
CUFLAG = -Xptxas -dlcm=ca -O2 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_53,code=sm_53 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_75,code=sm_80 \
	-gencode arch=compute_86,code=sm_86
#-----------------------------------------
cuda: $(CUCPPS)

compile: $(OBJS)

exe: $(OBJS)
	@g++ -g -pthread -m64 $(OBJS) \
	$(PRJLIB)/libmrcfile.a \
	$(PRJLIB)/libutil.a \
	$(PRJLIB)/libcuutil.a \
	$(PRJLIB)/libcuutilfft.a \
	-L$(CUDALIB) -L/usr/lib64 \
	-lcufft -lcudart -lcuda -lc -lm -lpthread \
	-o GCtfFind
	@echo GCtfFind has been generated.

%.cpp: %.cu
	@$(NVCC) -cuda $(CUFLAG) -I$(PRJINC) $< -o $@
	@echo $< has been compiled.

%.o: %.cpp
	@$(CC) $(CFLAG) -I$(PRJINC) -I$(CUDAINC) \
		$< -o $@
	@echo $< has been compiled.

clean:
	@rm -f $(OBJS) $(CUCPPS) *.h~ makefile~ GCtfFind
