#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace GCTFFind;

void CCudaHelper::CheckError(const char* pcLabel)
{
	cudaError_t err = cudaGetLastError();
	if(err == cudaSuccess) return;
	printf("%s: %s\n", pcLabel, cudaGetErrorString(err));
}
