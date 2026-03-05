#include "CUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static __global__ void mGCalcMean
(	float* gfImg,
        int iSizeX,
	int iPixels, // iSizeX * iSizeY
        int iPadX,
	float* gfMean
)
{	extern __shared__ float s_afSum[];
	float* s_afCount = &s_afSum[blockDim.x];
	//---------------------------
	float fSum = 0.0f, fCount = 0.0f;
	//---------------------------
	for (int i=threadIdx.x; i<iPixels; i+=blockDim.x) 
	{	int y = i / iSizeX;
		float fVal = gfImg[y * iPadX + i % iSizeX];
		//-------------------
		if(fVal < (float)-1e15) continue;
		fSum += fVal;
		fCount += 1;
        }
	s_afSum[threadIdx.x] = fSum;
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
	if (s_afCount[0] == 0) gfMean[0] = 0.0f;
	else gfMean[0] = s_afSum[0] / s_afCount[0];
}

static __global__ void mGCalcMeanStd
(       float* gfImg,
        int iSizeX,
        int iPixels, // iSizeX * iSizeY
        int iPadX,
        float* gfMeanStd
)
{       extern __shared__ float s_afSum1[];
	float* s_afSum2 = &s_afSum1[blockDim.x];
        float* s_afCount = &s_afSum2[blockDim.x];
        //---------------------------
        float fSum1 = 0.0f, fSum2 = 0.0f;
	float fCount = 0.0f;
        //---------------------------
        for (int i=threadIdx.x; i<iPixels; i+=blockDim.x)
        {       int y = i / iSizeX;
                float fVal = gfImg[y * iPadX + i % iSizeX];
                //-------------------
                if(fVal < (float)-1e15) continue;
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
        {       if (threadIdx.x < offset)
                {       int i = threadIdx.x + offset;
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


GCalcMeanStd::GCalcMeanStd(void)
{
	m_fMean = 0.0f;
	m_fStd = 0.0f;
}

GCalcMeanStd::~GCalcMeanStd(void)
{
}

float GCalcMeanStd::DoMean
(	float* gfImg,
	int* piImgSize,
	bool bPadded
)
{	m_fMean = 0.0f;
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
	int iSmBytes = aBlockDim.x * 2 * sizeof(float);
	//---------------------------
	float* gfMean = 0L;
	cudaMalloc(&gfMean, sizeof(float));
	//---------------------------
	mGCalcMean<<<aGridDim, aBlockDim, iSmBytes>>>(
	   gfImg, iImgX, iPixels, piImgSize[0], gfMean);
	//---------------------------
	cudaMemcpy(&m_fMean, gfMean, sizeof(float), cudaMemcpyDefault);
	if(gfMean != 0L) cudaFree(gfMean);
	return m_fMean;
}

float GCalcMeanStd::DoStd
(       float* gfImg,
        int* piImgSize,
        bool bPadded
)
{       m_fMean = 0.0f;
        m_fStd = 0.0f;
	//---------------------------
	int iImgX = piImgSize[0];
	if(bPadded) iImgX = (piImgSize[0] / 2 - 1) * 2;
	int iPixels = iImgX * piImgSize[1];
	//---------------------------
	dim3 aGridDim(1, 1);
	dim3 aBlockDim(1, 1);
	if(iPixels > 512) aBlockDim.x = 512;
	else if(iPixels > 256) aBlockDim.x = 256;
	else aBlockDim.x = 128;
	int iSmBytes = aBlockDim.x * 3 * sizeof(float);
	//---------------------------
	float* gfMeanStd = 0L;
	cudaMalloc(&gfMeanStd, 2 * sizeof(float));
	//---------------------------
	mGCalcMeanStd<<<aGridDim, aBlockDim, iSmBytes>>>(
	   gfImg, iImgX, iPixels, piImgSize[0], gfMeanStd);
	//---------------------------
	float afMeanStd[2] = {0.0f};
	cudaMemcpy(afMeanStd, gfMeanStd, 2 * sizeof(float), 
	   cudaMemcpyDefault);
	if(gfMeanStd != 0L) cudaFree(gfMeanStd);
	//---------------------------
	m_fMean = afMeanStd[0];
	m_fStd = afMeanStd[1];
	return m_fStd;
}

