#include "CFindCTFInc.h"
#include "../Util/CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace GCTFFind;

//-----------------------------------------------------------------------------
// 1. The zero-frequency component is at (x=0, y=iCmpY/2). The frequency
//    range in y direction is [-CmpY/2, CmpY/2).
// 2. fFreqLow, fFreqHigh are in the range of [0, 0.5f] of unit 1/pixel.
//-----------------------------------------------------------------------------
static __global__ void mGCalcCovar2D
(	float* gfSpectrum,
	int iCmpX,
	int iCmpY,
	float* gfRet
)
{	extern __shared__ float s_afCovar[];
	float* s_afVarX = &s_afCovar[blockDim.x];
	float* s_afVarY = &s_afVarX[blockDim.x];
	float* s_afWeight = &s_afVarY[blockDim.x];
	//---------------------------
	float fSumX2 = 0.0f;
	float fSumY2 = 0.0f;
	float fSumXY = 0.0f;
	float fSumW = 0.0f;
	//---------------------------
	int iHalfN = iCmpY / 2;
	int iCmpSize = iCmpX * iCmpY;
	//---------------------------
	for(int i=threadIdx.x; i<iCmpSize;  i+=blockDim.x)
	{	int x = i % iCmpX;
		int y = i / iCmpX - iHalfN;
		float fX = x / (float)iCmpY;
		float fY = (y - iHalfN) / (float)iCmpY;
		if(fabsf(fX) < 0.01f || fabsf(fY) < 0.01f) continue;
		//-------------------
		float fW = fabsf(gfSpectrum[y * iCmpX + x]);
		//-------------------
		fSumX2 += (fW * fX * fX);
		fSumY2 += (fW * fY * fY);
		fSumXY += (fW * fX * fY);
		fSumW += fW;
	}
	s_afCovar[threadIdx.x] = fSumXY;
	s_afVarX[threadIdx.x] = fSumX2;
	s_afVarY[threadIdx.x] = fSumY2;
	s_afWeight[threadIdx.x] = fSumW;
	__syncthreads();
	//---------------------------
	for(int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	int i = offset + threadIdx.x;
			s_afCovar[threadIdx.x] += s_afCovar[i];
			s_afVarX[threadIdx.x] += s_afVarX[i];
			s_afVarY[threadIdx.x] += s_afVarY[i];
			s_afWeight[threadIdx.x] += s_afWeight[i];
		}
		__syncthreads();
	}
	//---------------------------
	if(threadIdx.x != 0) return;
	gfRet[0] = s_afCovar[0] / s_afWeight[0];
	gfRet[1] = s_afVarX[0] / s_afWeight[0];
	gfRet[2] = s_afVarY[0] / s_afWeight[0];
}

GAstAngle::GAstAngle(void)
{
}

GAstAngle::~GAstAngle(void)
{
}

void GAstAngle::DoIt(float* gfSpectrum, int* piCmpSize)
{
	dim3 aBlockDim, aGridDim;
	aBlockDim = dim3(256, 1);
	aGridDim = dim3(1, 1);
	//---------------------------
	float* gfBuf = 0L;
	cudaMalloc(&gfBuf, sizeof(float) * 3);	
	size_t tSmBytes = sizeof(float) * aBlockDim.x * 4;
	//---------------------------
	mGCalcCovar2D<<<aGridDim, aBlockDim, tSmBytes>>>(
	   gfSpectrum, piCmpSize[0], piCmpSize[1],
	   gfBuf);
	//---------------------------
	float* pfCovar = new float[3];
	cudaMemcpy(pfCovar, gfBuf, sizeof(float) * 3,
	   cudaMemcpyDefault);
	if(gfBuf != 0L) cudaFree(gfBuf);
	//---------------------------
	mCalcEigens(pfCovar);
	if(pfCovar != 0L) delete[] pfCovar;
}

void GAstAngle::mCalcEigens(float* pfCovar)
{
	float a = pfCovar[1];
	float c = pfCovar[2];
	float b = pfCovar[0];  // covariance
	//---------------------------
	float fDelta = (float)sqrt((a - c) * (a - c) + 4.0 * b * b);
	float fLambda1 = ((a + c) + fDelta) * 0.5f;
	float fLambda2 = ((a + c) - fDelta) * 0.5f;
	//---------------------------
	float fX = b;
	float fY = fLambda1 - a;
	m_fAstAng = (float)atan(fY / (fX + 1e-30));
	m_fAstAng *= 57.296f;
}

