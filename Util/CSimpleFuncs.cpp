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

