#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static __global__ void mGRealResize2D
( 	float* gfInImg,
	int iInSizeX,
	int iInPadX,
	int iInSizeY,	
	float* gfOutImg,
	int iOutPadX,
	int iOutSizeY,
	bool bSum
)
{	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iOutSizeY) return;
	//---------------------------
	float fInX = blockIdx.x * iInSizeX / (float)gridDim.x;
	float fInY = y * iInSizeY / (float)iOutSizeY;
	//---------------------------
	int iInX = (int)fInX;
	int iInY = (int)fInY;
	float dx = fInX - iInX;
	float dy = fInY - iInY;
	if(iInX >= (iInSizeX - 1)) iInX = iInSizeX - 2;
	if(iInY >= (iInSizeY - 1)) iInY = iInSizeY - 2;
	//---------------------------
	int i = iInY * iInPadX + iInX;
	float fVal =
	   gfInImg[i] * dx * dy +
	   gfInImg[i + 1] * (1.0f - dx) * dy +
	   gfInImg[i + iInPadX] * dx * (1.0f - dy) +
	   gfInImg[i + iInPadX + 1] * (1.0f - dx) * (1.0f - dy);
	//---------------------------
	i = y * iOutPadX + blockIdx.x;
	if(bSum) gfOutImg[i] += fVal;
	else gfOutImg[i] = fVal;
}

GRealResize2D::GRealResize2D(void)
{
}

GRealResize2D::~GRealResize2D(void)
{
}

void GRealResize2D::GetNewSize
(	int* piInSize,
	bool bInPadded,
	float fBin,
	int* piOutSize,
	bool bOutPadded
)
{	int iInSizeX = bInPadded ? (piInSize[0] / 2 - 1) * 2 : piInSize[0];
	int iOutSizeX = (int)(iInSizeX / fBin + 0.5f) / 2 * 2;
	int iOutSizeY = (int)(piInSize[1] / fBin + 0.5f) / 2 * 2;
	piOutSize[0] = iOutSizeX;
	piOutSize[1] = iOutSizeY;
	if(bOutPadded) piOutSize[0] += 2;
}

float GRealResize2D::GetBinning
(	int* piInSize,
	bool bInPad,
	int* piOutSize,
	bool bOutPad
)
{	float fBin = piInSize[1] / (float)piOutSize[1];
	return fBin;	
}

void GRealResize2D::DoIt
( 	float* gfInImg, 
	int* piInSize,
	bool bInPadded,
  	float* gfOutImg,
	int* piOutSize,
	bool bOutPadded,
	bool bSum,
	cudaStream_t stream
)
{	int iInSizeX = piInSize[0];
	if(bInPadded) iInSizeX = (iInSizeX / 2 - 1) * 2;
	int iOutSizeX = piOutSize[0];
	if(bOutPadded) iOutSizeX = (iOutSizeX / 2 - 1) * 2;
	//---------------------------
	int iNumBlocks = piOutSize[1] / 32;
	if(iNumBlocks < 1) iNumBlocks = 1;
	else if(iNumBlocks > 16) iNumBlocks = 16;
	int iBlockSizeY = iNumBlocks * 32;
	//---------------------------
	dim3 aBlockDim(1, iBlockSizeY);
	dim3 aGridDim(iOutSizeX, 1);
	aGridDim.y = (piOutSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//---------------------------
	mGRealResize2D<<<aGridDim, aBlockDim, 0, stream>>>(
	   gfInImg, iInSizeX, piInSize[0], piInSize[1], 
	   gfOutImg, piOutSize[0], piOutSize[1], 
	   bSum);
	//---------------------------
	m_fBin = this->GetBinning(piInSize, bInPadded,
	   piOutSize, bOutPadded);	
}

