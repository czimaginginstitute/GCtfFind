#include "CUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static __global__ void mGCalcMeanStd
(       float* gfImg,
	float* gfRef,
	int iSizeX,
	int iPixels, // iSizeX * iSizeY
	int iPadX,
	float* gfMeanStd
)
{       extern __shared__ float s_afSum1[];
	float* s_afSum2 = &s_afSum1[blockDim.x];
	float* s_afCount = &s_afSum2[blockDim.x];
	//---------------------------
	float fSum1 = 0.0f; 
	float fSum2 = 0.0f;
	float fCount = 0.0f;
	//---------------------------
	for (int i=threadIdx.x; i<iPixels; i+=blockDim.x)
	{	int j = (i / iSizeX) * iPadX + i % iSizeX;
		float fVal = gfImg[j];
		float fRef = gfRef[j];
		//-------------------
		if(fVal < (float)-1e15) continue;
		else if(fRef < (float)-1e15) continue;
		//-------------------
		fSum1 += fVal;
		fSum2 += (fVal * fVal);
		fCount += 1;
	}
	s_afSum1[threadIdx.x] = fSum1;
	s_afSum2[threadIdx.x] = fSum2;
	s_afCount[threadIdx.x] = fCount;
	__syncthreads();
	//---------------------------
	for (int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if (threadIdx.x < offset)
		{	int i = threadIdx.x + offset;
			s_afSum1[threadIdx.x] += s_afSum1[i];
			s_afSum2[threadIdx.x] += s_afSum2[i];
			s_afCount[threadIdx.x] += s_afCount[i];
		}
		__syncthreads();
	}
	if (threadIdx.x != 0) return;
	if (s_afCount[0] == 0) 
	{	gfMeanStd[0] = 0.0f;
		gfMeanStd[1] = 0.0f;
	}
	else 
	{	float fMean = s_afSum1[0] / s_afCount[0];
		float fStd = s_afSum2[0] / s_afCount[0] - fMean * fMean;
		if(fStd < 0) fStd = 0.0f;
		else fStd = sqrtf(fStd);
		gfMeanStd[0] = fMean;
		gfMeanStd[1] = fStd;
	}
}

static __global__ void mGCovar
(	float* gfImg1,
	float* gfImg2,
        int iSizeX,
	int iPixels, // iSizeX * iSizeY
        int iPadX,
	float* gfCovar
)
{	extern __shared__ float s_afSum[];
	float* s_afCount = &s_afSum[blockDim.x];
	//---------------------------
	float fSum12 = 0.0f;
	float fCount = 0.0f;
	//---------------------------
	for (int i=threadIdx.x; i<iPixels; i+=blockDim.x) 
	{	int j = (i / iSizeX) * iPadX + i % iSizeX;
		float v1 = gfImg1[j];
		float v2 = gfImg2[j];
		//-------------------
		if(v1 < (float)-1e15) continue;
		else if(v2 < (float)-1e15) continue;
		//-------------------
		fSum12 += (v1 * v2);
		fCount += 1;
	}
	s_afSum[threadIdx.x] = fSum12;
	s_afCount[threadIdx.x] = fCount;
	__syncthreads();
	//---------------------------
	for (int offset=blockDim.x/2; offset>0; offset=offset/2) 
	{	if (threadIdx.x < offset)
		{	int i = threadIdx.x + offset;
			s_afSum[threadIdx.x] += s_afSum[i];
			s_afCount[threadIdx.x] += s_afCount[i];
		}
		__syncthreads();
	}
        if (threadIdx.x != 0) return;
	if (s_afCount[0] == 0) gfCovar[0] = 0.0f;
	else gfCovar[0] = s_afSum[0] / s_afCount[0];
}

GCalcCC2D::GCalcCC2D(void)
{
	m_fCC = 0.0f;
	memset(m_afMeanStd1, 0, sizeof(m_afMeanStd1));
	memset(m_afMeanStd2, 0, sizeof(m_afMeanStd2));
}

GCalcCC2D::~GCalcCC2D(void)
{
}

float GCalcCC2D::DoIt
(	float* gfImg1,
	float* gfImg2,
	int* piImgSize,
	bool bPadded
)
{	m_fCC = 0.0f;
	memset(m_afMeanStd1, 0, sizeof(float) * 2);
	memset(m_afMeanStd2, 0, sizeof(float) * 2);
	//---------------------------
	int iImgX = piImgSize[0];
	if(bPadded) iImgX = (piImgSize[0] / 2 - 1) * 2;
	int iPixels = iImgX * piImgSize[1];
	//---------------------------
	dim3 aGridDim(1, 1);
	dim3 aBlockDim(1, 1);
	if(iPixels > 1024) aBlockDim.x = 1024;
	else if(iPixels > 512) aBlockDim.x = 512;
	else if(iPixels > 256) aBlockDim.x = 256;
	else aBlockDim.x = 128;
	int iSmBytes = aBlockDim.x * 3 * sizeof(float);
	//---------------------------
	float* gfBuf = 0L;
	cudaMalloc(&gfBuf, sizeof(float) * 2);
	//---------------------------
	mGCalcMeanStd<<<aGridDim, aBlockDim, iSmBytes>>>(gfImg1, gfImg2,
	   iImgX, iPixels, piImgSize[0], gfBuf);
	cudaMemcpy(m_afMeanStd1, gfBuf, sizeof(float) * 2,
	   cudaMemcpyDefault);
	//---------------------------
	mGCalcMeanStd<<<aGridDim, aBlockDim, iSmBytes>>>(gfImg2, gfImg1,
	   iImgX, iPixels, piImgSize[0], gfBuf);
	cudaMemcpy(m_afMeanStd2, gfBuf, sizeof(float) * 2,
	   cudaMemcpyDefault);
	//---------------------------
	iSmBytes = aBlockDim.x * 2 * sizeof(float);
	mGCovar<<<aGridDim, aBlockDim, iSmBytes>>>(gfImg1, gfImg2,
	   iImgX, iPixels, piImgSize[0], gfBuf);
	cudaMemcpy(&m_fCC, gfBuf, sizeof(float), cudaMemcpyDefault);
	//---------------------------
	float fStd12 = m_afMeanStd1[1] * m_afMeanStd2[1];
	m_fCC = m_fCC - m_afMeanStd1[0] * m_afMeanStd2[0];
	if(fStd12 == 0) m_fCC = 0.0f;
	else m_fCC = m_fCC / fStd12;
	return m_fCC;
}

