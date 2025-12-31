#include "CFindCTFInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace GCTFFind;

//-----------------------------------------------------------------------------
// 1. Project intensity to the x axis. Each block calculuates the projection
//    at a single x point.
// 2. A block is 1D in y.
//-----------------------------------------------------------------------------
static __global__ void mGProjX
(	float* gfSpectrum,
	int iCmpX,
	int iCmpY,
	float* gfRet
)
{	extern __shared__ float s_afProjX[];
	float* s_afCount = &s_afProjX[blockDim.y];
	//---------------------------
	float fSum = 0.0f;
	float fCount = 0.0f;
	//---------------------------
	int iHalfX = iCmpX - 1;
	int iHalfY = iCmpY / 2;
	int x = (int)blockIdx.x - iHalfX;
	//---------------------------
	int iSign = (x < 0) ? -1 : 1;
	x *= iSign;
	//---------------------------
	for(int i=threadIdx.y; i<iCmpY; i+=blockDim.y)
	{	int y = (i - iHalfY) * iSign;
		if(y < 0) y += iCmpY;
		fSum += fabsf(gfSpectrum[y * iCmpX + x]);
		fCount += 1;
	}	
	s_afProjX[threadIdx.y] = fSum;
	s_afCount[threadIdx.y] = fCount;
	__syncthreads();
	//---------------------------
	for(int offset=blockDim.y/2; offset>0; offset=offset/2)
	{	if(threadIdx.y < offset)
		{	int i = offset + threadIdx.y;
			s_afProjX[threadIdx.y] += s_afProjX[i];
			s_afCount[threadIdx.y] += s_afCount[i];
		}
		__syncthreads();
	}
	//---------------------------
	if(threadIdx.y != 0) return;
	if(s_afCount[0] == 0.0f) gfRet[blockIdx.x] = 0.0f;
	else gfRet[blockIdx.x] = s_afProjX[0] / s_afCount[0];
}

static __global__ void mGProjY
(       float* gfSpectrum,
        int iCmpX,
        int iCmpY,
        float* gfRet
)
{       extern __shared__ float s_afProjY[];
        float* s_afCount = &s_afProjY[blockDim.x];
        //---------------------------
        float fSum = 0.0f;
        float fCount = 0.0f;
        //---------------------------
        int iHalfX = iCmpX - 1;
	int iNx = iHalfX * 2;
	//---------------------------
        for(int i=threadIdx.x; i<iNx; i+=blockDim.x)
        {       int x = i - iHalfX;
		int y = (int)blockIdx.y;
		if(x < 0)
		{	x = -x;
			y = -y;
		}
		if(y < 0) y += iCmpY;
		//-------------------
                fSum += fabsf(gfSpectrum[y * iCmpX + x]);
                fCount += 1;
        }
        s_afProjY[threadIdx.x] = fSum;
        s_afCount[threadIdx.x] = fCount;
        __syncthreads();
	//---------------------------
        for(int offset=blockDim.x/2; offset>0; offset=offset/2)
        {       if(threadIdx.x < offset)
                {       int i = offset + threadIdx.x;
                        s_afProjY[threadIdx.x] += s_afProjY[i];
                        s_afCount[threadIdx.x] += s_afCount[i];
                }
                __syncthreads();
        }
        //---------------------------
        if(threadIdx.x != 0) return;
        if(s_afCount[0] == 0.0f) gfRet[blockIdx.y] = 0.0f;
        else gfRet[blockIdx.y] = s_afProjY[0] / s_afCount[0];
}

static __global__ void mGCalcCovar
(	float* gfProjX,
	float* gfProjY,
	int iSize,
	float* gfCovar
)
{	extern __shared__ float s_afSumX[];
	float* s_afSumY = &s_afSumX[blockDim.x];
	float* s_afSumX2 = &s_afSumY[blockDim.x];
	float* s_afSumY2 = &s_afSumX2[blockDim.x];
	float* s_afSumXY = &s_afSumY2[blockDim.x];
	//---------------------------
	float fSumX = 0.0f, fSumY = 0.0f;
	float fSumX2 = 0.0f, fSumY2 = 0.0f, fSumXY = 0.0f;
	for(int i=threadIdx.x; i<iSize; i+=blockDim.x)
	{	float fW = fabsf(i / (float)iSize - 0.5f);
		fW = fmaxf(fW - 0.25f, 0.0f);
		fW = expf(-10.0f * fW * fW);
		float fX = gfProjX[i] * fW;
		float fY = gfProjY[i] * fW;
		fSumX += fX;
		fSumY += fY;
		fSumX2 += (fX * fX);
		fSumY2 += (fY * fY);
		fSumXY += (fX * fY);
	}
	s_afSumX[threadIdx.x] = fSumX;
	s_afSumY[threadIdx.x] = fSumY;
	s_afSumX2[threadIdx.x] = fSumX2;
	s_afSumY2[threadIdx.x] = fSumY2;
	s_afSumXY[threadIdx.x] = fSumXY;
	__syncthreads();
	//---------------------------
	for(int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	int i = offset + threadIdx.x;
			s_afSumX[threadIdx.x] += s_afSumX[i];
			s_afSumY[threadIdx.x] += s_afSumY[i];
			s_afSumX2[threadIdx.x] += s_afSumX2[i];
			s_afSumY2[threadIdx.x] += s_afSumY2[i];
			s_afSumXY[threadIdx.x] += s_afSumXY[i];
		}
		__syncthreads();
	}
	//---------------------------
	if(threadIdx.x != 0) return;
	fSumX = s_afSumX[0] / iSize;
	fSumY = s_afSumY[0] / iSize;
	gfCovar[0] = s_afSumX2[0] / iSize - fSumX * fSumX;
	gfCovar[1] = s_afSumXY[0] / iSize - fSumX * fSumY;
	gfCovar[2] = s_afSumY2[0] / iSize - fSumY * fSumY;
}

GAstRatio::GAstRatio(void)
{
}

GAstRatio::~GAstRatio(void)
{
}

void GAstRatio::DoIt(float* gfSpectrum, int* piCmpSize)
{	
	int iNx = (piCmpSize[0] - 1) * 2;
	float* gfProjX = 0L;
	int iBytes = (iNx + piCmpSize[1] + 3) * sizeof(float);
	cudaMalloc(&gfProjX, iBytes);
	float* gfProjY = &gfProjX[iNx];
	float* gfCovar = &gfProjY[piCmpSize[1]];
	//---------------------------
	dim3 aBlockDim(1, 512); // y is projection direction
	dim3 aGridDim(iNx, 1);
	size_t tSmBytes = sizeof(float) * aBlockDim.y * 2;
	//---------------------------
	mGProjX<<<aGridDim, aBlockDim, tSmBytes>>>(gfSpectrum, 
	   piCmpSize[0], piCmpSize[1], gfProjX);
	//---------------------------
	aBlockDim = dim3(512, 1);
	aGridDim = dim3(1, piCmpSize[1]);
	tSmBytes = sizeof(float) * aBlockDim.x * 2;
	//---------------------------
	mGProjY<<<aGridDim, aBlockDim, tSmBytes>>>(gfSpectrum,
	   piCmpSize[0], piCmpSize[1], gfProjY);
	//---------------------------
	aBlockDim = dim3(256, 1);
	aGridDim = dim3(1, 1);
	tSmBytes = sizeof(float) * aBlockDim.x * 5;
	mGCalcCovar<<<aGridDim, aBlockDim, tSmBytes>>>(gfProjX, 
	   gfProjY, piCmpSize[1], gfCovar);
	//---------------------------
	float afCovar[3] = {0.0f};
	cudaMemcpy(afCovar, gfCovar, sizeof(float) * 3, cudaMemcpyDefault);
	mCalcEigens(afCovar);
	//---------------------------
	if(gfProjX != 0L) cudaFree(gfProjX);
}

void GAstRatio::mCalcEigens(float* pfCovar)
{
	float a = pfCovar[0];
	float b = pfCovar[1];
	float c = pfCovar[2];
	//---------------------------
	float fDelta = (float)sqrt((a - c) * (a - c) + 4.0 * b * b);
	float fLambda1 = ((a + c) + fDelta) * 0.5f;
	float fLambda2 = ((a + c) - fDelta) * 0.5f;
	//---------------------------
	m_fAstRatio = fLambda2 / fLambda1;
	//printf("Covar matrix: %.4e  %.4e  %.4e\n", pfCovar[0],
	//	pfCovar[1], pfCovar[2]);
	//printf("lambda: %.4e  %.4e\n", fLambda1, fLambda2);
	//printf("Astigmatism: %.4e  %.4e\n\n", m_fAstRatio);
	
}

