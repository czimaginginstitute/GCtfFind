#include "CFindCTFInc.h"
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
	float fPower,
	float* gfRet
)
{	extern __shared__ float s_afCovar[];
	float* s_afCount = &s_afCovar[blockDim.x];
	//---------------------------
	float fSum = 0.0f;
	float fCount = 0.0f;
	//---------------------------
	int iHalfN = iCmpY / 2;
	int iCmpSize = iCmpY * iCmpY;
	//---------------------------
	for(int i=threadIdx.x; i<iCmpSize;  i+=blockDim.x)
	{	int x = i % iCmpY - iHalfN;
		int y = i / iCmpY - iHalfN;
		float fX = x / (float)iCmpY;
		float fY = y / (float)iCmpY;
		//-------------------
		int iSign = (x < 0) ? -1 : 1;
		x *= iSign;
		y *= iSign;
		fY *= iSign;
		if(y < 0) y += iCmpY;
		//-------------------
		float fX2 = fX * fX;
		float fY2 = fY * fY;
		//-------------------
		float fInt = gfSpectrum[y * iCmpX + x] / iCmpSize;
		fInt = powf(fabsf(fInt), fPower);
		if(blockIdx.x == 0) fSum += (fX * fY * fInt);
		else if(blockIdx.x == 1) fSum += (fX2 * fInt);
		else if(blockIdx.x == 2) fSum += (fY2 * fInt);
		fCount += fInt;

	}
	s_afCovar[threadIdx.x] = fSum;
	s_afCount[threadIdx.x] = fCount;
	__syncthreads();
	//---------------------------
	for(int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	int i = offset + threadIdx.x;
			s_afCovar[threadIdx.x] += s_afCovar[i];
			s_afCount[threadIdx.x] += s_afCount[i];
		}
		__syncthreads();
	}
	//---------------------------
	if(threadIdx.x != 0) return;
	if(s_afCount[0] == 0.0f) gfRet[blockIdx.x] = 0.0f;
	else gfRet[blockIdx.x] = s_afCovar[0] / s_afCount[0];
}

static __global__ void mGCalcPower
(	float* gfSpect,
	int iCmpX,
	int iCmpY,
	float* gfRet
)
{	extern __shared__ float s_afSumX[];
	float* s_afSumY = &s_afSumX[blockDim.x];
	//---------------------------
	float fSumX = 0.0f, fSumY = 0.0f;
	int iSize = iCmpX * iCmpY;
	for(int i=threadIdx.x; i<iSize; i+=blockDim.x)
	{	float fX = (i % iCmpX) / (float)iCmpY;
		float fY = (i / iCmpY) / (float)iCmpY;
		if(fY > 0.5f) fY -= 1.0f;
		//-------------------
		fSumX += (fX * fX + fY * fY);
		fSumY += logf(fabs(gfSpect[i] / iSize) + (float)1e-10);
	}
	s_afSumX[threadIdx.x] = fSumX;
	s_afSumY[threadIdx.x] = fSumY;
	__syncthreads();
	//---------------------------
	for(int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	int i = offset + threadIdx.x;
			s_afSumX[threadIdx.x] += s_afSumX[i];
			s_afSumY[threadIdx.x] += s_afSumY[i];
		}
		__syncthreads();
	}
	//---------------------------
	if(threadIdx.x != 0) return;
	float fPower = 1.0f;
	if(s_afSumY[0] > 0) fPower = s_afSumX[0] / s_afSumY[0] * 0.5f;
	gfRet[0] = fPower;
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
	aBlockDim = dim3(512, 1);
	aGridDim = dim3(3, 1);
	//---------------------------
	float* gfBuf = 0L;
	cudaMalloc(&gfBuf, aGridDim.x * sizeof(float));	
	size_t tSmBytes = sizeof(float) * aBlockDim.x * 2;
	//---------------------------
	float fPower = 0.01f;
	mGCalcPower<<<dim3(1, 1), aBlockDim, tSmBytes>>>(gfSpectrum,
	   piCmpSize[0], piCmpSize[1], gfBuf);
	cudaMemcpy(&fPower, gfBuf, sizeof(float), cudaMemcpyDefault);
	//---------------------------
	mGCalcCovar2D<<<aGridDim, aBlockDim, tSmBytes>>>(gfSpectrum, 
	   piCmpSize[0], piCmpSize[1], fPower, gfBuf);
	//---------------------------
	float* pfCovar = new float[aGridDim.x];
	cudaMemcpy(pfCovar, gfBuf, aGridDim.x * sizeof(float),
	   cudaMemcpyDefault);
	if(gfBuf != 0L) cudaFree(gfBuf);
	//---------------------------
	mCalcEigens(pfCovar);
	if(pfCovar != 0L) delete[] pfCovar;
	//printf("fPower = %e\n", fPower);
}

void GAstAngle::mCalcEigens(float* pfCovar)
{
	float a = fmaxf(pfCovar[1], pfCovar[2]);
	float c = fminf(pfCovar[1], pfCovar[2]);
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

	//printf("Covar matrix: %.4e  %.4e  %.4e\n", pfCovar[0],
	//	pfCovar[1], pfCovar[2]);
	//printf("Astigmatism: %.4e  %.4e\n\n", m_fAstRatio, m_fAstAng);
}

