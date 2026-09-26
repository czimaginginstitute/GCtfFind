#include "CLppInc.h"
#include "../Util/CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

//--------------------------------------------------------------------
// 1. Calculate the halfp hase spectrum in radian.
// 2. The DC is shifted to (0, iCmpY/2).
//--------------------------------------------------------------------
static __global__ void mGHalfPhase
(	cufftComplex* gCmp,
	float* gfPhase,
	int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//--------------------------------------------
	// put DC at x = 0, y = iCmpY / 2
	//--------------------------------------------
	y = y + iCmpY / 2;
	if(y >= iCmpY) y = y - iCmpY;
	int j = y * gridDim.x + blockIdx.x;
	//---------------------------
	float fRe = gCmp[i].x;
	float fIm = gCmp[i].y;
	if(fRe == 0.0f)
	{	if(fIm > 0) gfPhase[j] = 1.5708f;
		else if(fIm < 0) gfPhase[j] = -1.5708f;
		else gfPhase[j] = 0.0f;
	}
	else gfPhase[j] = atanf(fIm / fRe);
}

//--------------------------------------------------------------------
// 1. DC of gfHalfPhase is already at (0, iCmpY / 2)
// 2. DC of gfFullPhase will be at (iCmpX / 2, iCmpY / 2)
// 3. The phases in negative x range are negative mirror of the 
//    positive ones w.r.t. the DC.
//--------------------------------------------------------------------
static __global__ void mGFullPhase
(	float* gfHalfPhase,
	float* gfFullPhase,
	int iHalfX,
	int iCmpY,
	int iFullSizeX
)
{	int xSrc, ySrc, xDst, yDst;
	yDst = blockIdx.y * blockDim.y + threadIdx.y;
	if(yDst >= iCmpY) return;
	//-----------------------
	xDst = blockIdx.x - iHalfX;
	int iSign = (xDst >= 0) ? 1 : -1;
	xSrc = iSign * xDst;
	ySrc = (iCmpY + iSign * yDst) % iCmpY;
	//-----------------------
	gfFullPhase[yDst * iFullSizeX + blockIdx.x] = 
	   iSign * gfHalfPhase[ySrc * (iHalfX + 1) + xSrc];
}

GCalcPhase2D::GCalcPhase2D(void)
{
}

GCalcPhase2D::~GCalcPhase2D(void)
{
}

void GCalcPhase2D::DoHalf
(	cufftComplex* gCmp, 
	float* gfHalfPhase, 
	int* piCmpSize
)
{	dim3 aBlockDim(1, 512);
	int iGridY = piCmpSize[1] / aBlockDim.y + 1;
	dim3 aGridDim(piCmpSize[0], iGridY);
	mGHalfPhase<<<aGridDim, aBlockDim>>>(
	   gCmp, gfHalfPhase, piCmpSize[1]);
}

void GCalcPhase2D::DoFull
(	float* gfHalfPhase, 
	int* piCmpSize,
        float* gfFullPhase,
	bool bFullPadded
)
{	int iHalfX = piCmpSize[0] - 1;
	int iNx = iHalfX * 2;
	int iFullSizeX = bFullPadded ? iNx + 2 : iNx;
	//-------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iNx, 1);
	aGridDim.y = (piCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//----------------------------------------------------------
	mGFullPhase<<<aGridDim, aBlockDim>>>(gfHalfPhase,
	   gfFullPhase, iHalfX, piCmpSize[1], iFullSizeX);
}	
