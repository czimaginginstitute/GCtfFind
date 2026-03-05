#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

using namespace GCTFFind;

void CSimpleFuncs::CheckCudaError(const char* pcLocation)
{
	cudaError_t cuErr = cudaGetLastError();
	if(cuErr == cudaSuccess) return;
	//---------------------------
	fprintf(stderr, "%s: %s\n\t\n\n", pcLocation,
		cudaGetErrorString(cuErr));
	cudaDeviceReset();
	assert(0);
}

float* CSimpleFuncs::GAllocFloat(int* piSize)
{
	int iSize = piSize[0] * piSize[1];
	float* gfBuf = CSimpleFuncs::GAllocFloat(iSize);
	return gfBuf;
}

float* CSimpleFuncs::GAllocFloat(int iSize)
{
	float* gfBuf = 0L;
	int tBytes = sizeof(float) * iSize;
	if(tBytes == 0) return 0L;
	cudaMalloc(&gfBuf, tBytes);
	return gfBuf;
}

cufftComplex* CSimpleFuncs::GAllocCmp(int* piSize)
{
	int iSize = piSize[0] * piSize[1];
	cufftComplex* gcmpBuf = CSimpleFuncs::GAllocCmp(iSize);
	return gcmpBuf;
}

cufftComplex* CSimpleFuncs::GAllocCmp(int iSize)
{
	cufftComplex* gcmpBuf = 0L;
	size_t tBytes = sizeof(cufftComplex) * iSize;
	if(tBytes <= 0) return 0L;
	cudaMalloc(&gcmpBuf, tBytes);
	return gcmpBuf;
}
