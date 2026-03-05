#include "CLppInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

//-----------------------------------------------------------------------------
// 1. Search twin peaks in the Fourier transform of xLPP laser fringes. The
//    initial observation is the twin peaks of these fringes are close to
//    90 degrees apart and have approximately the same length, i.e. they
//    are conjugated.
// 2. For each point in the positive frequency range, calculate the sum of
//    two amplitudes, one at the search point, the other at the conjugated
//    point.
// 3. Since the conjugation may not be perfect in terms of length and angle,
//    we search in a small circular range centered at the nominal conjugated
//    point.
// 4. piPeaks stores the index of the conjugated point for each positive
//    frequency.
// 5. pfPeaks stores the amplitude sum for each pair of the conjugated points.
//  
//-----------------------------------------------------------------------------
static __global__ void mGCalcTwinPeaks
(	cufftComplex* gCmp,
	int iCmpX,  int iCmpY,
	int iMaskR, int iSearR,
	int* giPeaks, float* gfPeaks
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= (iCmpY / 2)) return;
	//---------------------------
	int i = y * iCmpX + blockIdx.x;
	giPeaks[i] = 0;
	gfPeaks[i] = (float)-1e20;
	//---------------------------
	float fTmp1 = 0.0f, fTmp2 = 0.0f;
	fTmp1 = sqrtf(blockIdx.x * blockIdx.x + y * y);
	if(fTmp1 <= iMaskR) return;
	//---------------------------
	int xc = y;
	int yc = iCmpY - blockIdx.x;
	int iPeak2 = 0;
	float fPeak2 = (float)-1e20;
	//---------------------------
	for(int j=-iSearR; j<=iSearR; j++)
	{	y = yc + j;
		if(y < 0 || y >= iCmpY) continue;
		for(int k=-iSearR; k<=iSearR; k++)
		{	int x = xc + k;
			if(x < 0 || x >= gridDim.x) continue;
			int m = y * iCmpX + x;
			//-----------
			fTmp1 = gCmp[m].x;
			fTmp2 = gCmp[m].y;
			fTmp1 = fTmp1 * fTmp1 + fTmp2 * fTmp2;
			if(fTmp1 > fPeak2)
			{	fPeak2 = fTmp1;
				iPeak2 = m;
			}
		}
	}
	//---------------------------
	fTmp1 = gCmp[i].x;
	fTmp2 = gCmp[i].y;
	fTmp1 = fTmp1 * fTmp1 + fTmp2 * fTmp2;
	//---------------------------
	giPeaks[i] = iPeak2;
	gfPeaks[i] = sqrtf(fTmp1 + fPeak2);
}

static __global__ void mGFindPeak
(	int* giLocs,
	float* gfPeaks,
	int iSize
)
{	extern __shared__ float s_afPeaks[];
	float* s_afLocs = &s_afPeaks[blockDim.x];
	//---------------------------
	float fLoc = 0.0f;
	float fPeak = (float)-1e20;
	//---------------------------
	for(int i=threadIdx.x; i<iSize; i+=blockDim.x)
	{	float fAmp = gfPeaks[i];
		if(fAmp > fPeak)
		{	fPeak = fAmp;
			fLoc = i;
		}
	}
	s_afPeaks[threadIdx.x] = fPeak;
	s_afLocs[threadIdx.x] = fLoc;
	//---------------------------
	for(int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	int i = threadIdx.x + offset;
			if(s_afPeaks[threadIdx.x] < s_afPeaks[i])
			{	s_afPeaks[threadIdx.x] = s_afPeaks[i];
				s_afLocs[threadIdx.x] = s_afLocs[i];
			}
			__syncthreads();
		}
	}
	//---------------------------
	if(threadIdx.x != 0) return;
	giLocs[iSize] = (int)(s_afLocs[0] + 0.1f);
	gfPeaks[iSize] = s_afPeaks[0];
}

GFindTwinPeaks::GFindTwinPeaks(void)
{
	m_gfPeaks = 0L;
	m_giLocs = 0L;
}

GFindTwinPeaks::~GFindTwinPeaks(void)
{
	this->Clean();
}

void GFindTwinPeaks::Clean(void)
{
	if(m_gfPeaks != 0L)
	{	cudaFree(m_gfPeaks);
		m_gfPeaks = 0L;
	}
	if(m_giLocs != 0L)
	{	cudaFree(m_giLocs);
		m_giLocs = 0L;
	}
}

void GFindTwinPeaks::SetSize(int* piCmpSize)
{
	this->Clean();
	//---------------------------
	m_aiCmpSize[0] = piCmpSize[0];
	m_aiCmpSize[1] = piCmpSize[1];
	m_aiHalfSize[0] = m_aiCmpSize[0] - 1;
	m_aiHalfSize[1] = m_aiCmpSize[1] / 2;
	//---------------------------
	int iSize = m_aiCmpSize[0] * m_aiHalfSize[1] + 1;
	cudaMalloc(&m_gfPeaks, iSize * sizeof(float));
	cudaMalloc(&m_giLocs, iSize * sizeof(int));
}

void GFindTwinPeaks::DoIt(cufftComplex* gCmp)
{	
	mCalcPeaks(gCmp);
	mFindTwinPeaks(gCmp);
}

void GFindTwinPeaks::mCalcPeaks(cufftComplex* gCmp)
{
	int iMaskR = 5;
	int iSearR = 5;
	//---------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiHalfSize[0], 1);
	aGridDim.y = (m_aiHalfSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//---------------------------
	mGCalcTwinPeaks<<<aGridDim, aBlockDim>>>(
	   gCmp, 
	   m_aiCmpSize[0], m_aiCmpSize[1],
	   iMaskR, iSearR,
	   m_giLocs,
	   m_gfPeaks);
}

void GFindTwinPeaks::mFindTwinPeaks(cufftComplex* gCmp)
{
	int iSize = m_aiCmpSize[0] * m_aiHalfSize[1];
	//---------------------------
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	int ismBytes = aBlockDim.x * sizeof(float) * 2;
	//---------------------------
	mGFindPeak<<<aGridDim, aBlockDim, ismBytes>>>(
	   m_giLocs, m_gfPeaks, iSize);
	//---------------------------
	int iBytes = sizeof(int);
	cudaMemcpy(&m_iPeak1, &m_giLocs[iSize], iBytes, cudaMemcpyDefault);
	cudaMemcpy(&m_iPeak2, &m_giLocs[m_iPeak1], iBytes, cudaMemcpyDefault);
	//---------------------------
	iBytes = sizeof(cufftComplex);
	cudaMemcpy(&m_cmpPeak1, &gCmp[m_iPeak1], iBytes, cudaMemcpyDefault);
	cudaMemcpy(&m_cmpPeak2, &gCmp[m_iPeak2], iBytes, cudaMemcpyDefault);

}
