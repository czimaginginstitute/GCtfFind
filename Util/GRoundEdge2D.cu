#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static __global__ void mGRoundEdge2D
(	float* gfImg, 
	int iPadX, 
	int iSizeY,
	float fMaskSizeX, 
	float fMaskSizeY,
	float fBFactor
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * iPadX + blockIdx.x;
	//---------------------------
	float fX = blockIdx.x - gridDim.x * 0.5f;
	float fY = y - iSizeY * 0.5;
	float fW = sqrtf(fX * fX + fY * fY);
	if(fW <= fMaskSizeX) return;
	//---------------------------
	fX = (fW - fMaskSizeX) / gridDim.x;
	fY = (fW - fMaskSizeX) / iSizeY;
	fW = fX * fX + fY * fY;	
	fW = expf(-fBFactor * fW);
	gfImg[i] *= fW;
}

GRoundEdge2D::GRoundEdge2D(void)
{
	memset(m_afMaskSize, 0, sizeof(m_afMaskSize));
}

GRoundEdge2D::~GRoundEdge2D(void)
{
}

void GRoundEdge2D::SetMask(float fSizeX, float fSizeY)
{
	m_afMaskSize[0] = fSizeX;
	m_afMaskSize[1] = fSizeY;
}

void GRoundEdge2D::DoIt
(	float* gfImg, 
	int* piSize, 
	bool bPadded,
	float fBFactor
)
{	int iImgX = piSize[0];
	if(bPadded) iImgX = (piSize[0] / 2 - 1) * 2;
	//---------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iImgX, (piSize[1] + aBlockDim.y - 1) / aBlockDim.y);
	//---------------------------
	mGRoundEdge2D<<<aGridDim, aBlockDim>>>(gfImg, piSize[0], piSize[1],
	   m_afMaskSize[0], m_afMaskSize[1], fBFactor);
}

