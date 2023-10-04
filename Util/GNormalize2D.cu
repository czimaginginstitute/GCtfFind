#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static __global__ void mGNorm2D
(	float* gfImg,
	float fMean,
	float fStd,
	int iPadX,
	int iSizeY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * iPadX + blockIdx.x;
	float fInt = gfImg[i];
	if(fInt > (float)-1e20)
	{	gfImg[i] = (fInt - fMean) / fStd;
	}
}

GNormalize2D::GNormalize2D(void)
{
}

GNormalize2D::~GNormalize2D(void)
{
}

void GNormalize2D::DoIt
(	float* gfImg,
	float fMean,
	float fStd,
	int* piImgSize,
	bool bPadded
)
{	int iSizeX = piImgSize[0];
	if(bPadded) iSizeX = (piImgSize[0] / 2 - 1) * 2;
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iSizeX, 1);
	aGridDim.y = (piImgSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	mGNorm2D<<<aGridDim, aBlockDim>>>(gfImg, fMean, fStd,
	   piImgSize[0], piImgSize[1]);

}
