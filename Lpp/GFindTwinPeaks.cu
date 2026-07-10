#include "CLppInc.h"
#include "../CMainInc.h"
#include "../FindCTF/CFindCTFInc.h"
#include "../Util/CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace GCTFFind;

//-----------------------------------------------------------------------------
// 1. Rotation sum is calculated by rotating the positive x and negative y
//    Fourier components +90 degree and adding it the corresponding positive
//    x and positive y Fourier component.
// 2. The sum is amplitude sum.
//-----------------------------------------------------------------------------
static __global__ void mGCalcRotSum
(	float* gfHalfSpect,
	float* gfRotSum,
	int iCmpX,
	int iHalfY  // iHalfY = iCmpY / 2
)
{	int x = blockIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iHalfY) return;
	//---------------------------
	int i = y * iCmpX + x;
	//---------------------------
	// positive y frequency
	//---------------------------
	int j = (y + iHalfY) * iCmpX + x;
	gfRotSum[i] = gfHalfSpect[j];
	//------------------------------
	// rotate x and y by -90 degree
	//------------------------------
	int xRot = y;
	int yRot = -x + iHalfY;
	j = yRot * iCmpX + xRot;
	//---------------------------
	float fW = 1.0f - expf(-0.1f * (x * x + y * y));
	gfRotSum[i] = (gfRotSum[i] + gfHalfSpect[j]) * fW 
	   / (iCmpX * iHalfY);
}

static __global__ void mGFindPeak
(	float* gfRotSum,
	int iCmpX,
	int iHalfY,
	float fMaskR,
	float* gfPeakLoc
)
{	extern __shared__ float s_afPeaks[];
	float* s_afLocs = &s_afPeaks[blockDim.x];
	//---------------------------
	float fLoc = 0.0f;
	float fPeak = (float)-1e20;
	int iSize = iCmpX * iHalfY;
	//---------------------------
	for(int i=threadIdx.x; i<iSize; i+=blockDim.x)
	{	int x = i % iCmpX;
		int y = i / iCmpX;
		float fR = sqrtf(x * x + y * y);
		if(fR < fMaskR) continue;
		//-------------------
		float fVal = gfRotSum[i];
		if(fVal <= fPeak) continue;
		fPeak = fVal;
		fLoc = i;
	}
	s_afPeaks[threadIdx.x] = fPeak;
	s_afLocs[threadIdx.x] = fLoc;
	__syncthreads();
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
	gfPeakLoc[0] = s_afLocs[0];
}

static __global__ void mGLocalSearch
(	float* gfHalfSpect,
	int iCmpX,
	int iCmpY,
	int iCentX,
	int iCentY,
	float fRadius,
	float* gfPeakLoc
)
{	extern __shared__ float s_afPeaks[];
	float* s_afLocs = &s_afPeaks[blockDim.x];
	//---------------------------
	float fLoc = 0.0f;
	float fPeak = (float)-1e20;
	int iSize = iCmpX * iCmpY;
	//---------------------------
	for(int i=threadIdx.x; i<iSize; i+=blockDim.x)
	{	int x = i % iCmpX - iCentX;
		int y = i / iCmpX - iCentY;
		if(x < 0 || y < 0) continue;
		//-------------------
		float fR = sqrtf(x * x + y * y);
		if(fR >= fRadius) continue;
		//-------------------
		float fVal = gfHalfSpect[i];
		if(fVal < fPeak) continue;
		//-------------------
		fPeak = fVal;
		fLoc = i;
	}
	s_afPeaks[threadIdx.x] = fPeak;
	s_afLocs[threadIdx.x] = fLoc;
	__syncthreads();
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
	gfPeakLoc[0] = s_afLocs[0];	
}

static void gSaveMrc(float* gfImg, int* piSize, const char* pcSuffix)
{
	CInput* pInput = CInput::GetInstance();
        char acOutMrc[256] = {'\0'};
        pInput->GetOutFile("Test", pcSuffix, acOutMrc);
        //---------------------------
        CSaveTempMrc saveMrc;
	char acExt[16] = {'\0'};
        saveMrc.SetFile(acOutMrc, acExt);
        saveMrc.GDoIt(gfImg, piSize);
}

GFindTwinPeaks::GFindTwinPeaks(void)
{
	m_pCufft2D = 0L;
	m_gfHalfSpect = 0L;
	m_gfRotSum = 0L;
	m_aiPadSize[0] = 0;
	m_aiPadSize[1] = 0;
	m_fPixSize = 1.0f;  // 1.0 Angstrom
}

GFindTwinPeaks::~GFindTwinPeaks(void)
{
	this->Clean();
}

void GFindTwinPeaks::Clean(void)
{
	if(m_gfHalfSpect != 0L) cudaFree(m_gfHalfSpect);
	if(m_gfRotSum != 0L) cudaFree(m_gfRotSum);
	if(m_pCufft2D != 0L) delete m_pCufft2D;
	m_gfHalfSpect = 0L;
	m_gfRotSum = 0L;
	m_pCufft2D = 0L;
}

void GFindTwinPeaks::SetPadSize(int* piPadSize)
{
	if(m_aiPadSize[0] != piPadSize[0]) this->Clean();
	else if(m_aiPadSize[1] != piPadSize[1]) this->Clean();
	else return;
	//---------------------------
	m_aiPadSize[0] = piPadSize[0];
	m_aiPadSize[1] = piPadSize[1];
	m_aiCmpSize[0] = m_aiPadSize[0] / 2;
	m_aiCmpSize[1] = m_aiPadSize[1];
	//---------------------------
	m_pCufft2D = new CCufft2D;
	bool bPadded = true;
	m_pCufft2D->CreateForwardPlan(m_aiPadSize, bPadded);
	//---------------------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gfHalfSpect, iCmpSize * sizeof(float));
	//---------------------------
	int iHalfSize = m_aiCmpSize[0] * (m_aiCmpSize[1] / 2);	
	cudaMalloc(&m_gfRotSum, iHalfSize * sizeof(float));
}

void GFindTwinPeaks::SetPixSize(float fPixSize)
{
	m_fPixSize = fPixSize;
}

void GFindTwinPeaks::DoIt(float* gfPadSpect)
{
	m_gfPadSpect = gfPadSpect;
	m_pCufft2D->Forward(m_gfPadSpect);
	cudaStreamSynchronize((cudaStream_t)0);
	//---------------------------
	GCalcSpectrum calcSpectrum;
	calcSpectrum.DoIt((cufftComplex*)m_gfPadSpect,
	   m_gfHalfSpect, m_aiCmpSize);
	//---------------------------
	mCalcRotSum();
	mFindAvgPeak();
	mFindTwinPeaks();
}

void GFindTwinPeaks::mCalcRotSum(void)
{
	int iHalfX = m_aiCmpSize[0] - 1;
	int iHalfY = m_aiCmpSize[1] / 2;
	//---------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iHalfX, 1);
	aGridDim.y = (iHalfY + aBlockDim.y - 1) / aBlockDim.y;
	//---------------------------
	mGCalcRotSum<<<aGridDim, aBlockDim>>>(m_gfHalfSpect,
	   m_gfRotSum, m_aiCmpSize[0], iHalfY);

	int aiSize[] = {m_aiCmpSize[0], iHalfY};
	gSaveMrc(m_gfRotSum, aiSize, "_RotSum.mrc");
}

void GFindTwinPeaks::mFindAvgPeak(void)
{
	int iHalfY = m_aiCmpSize[1] / 2;
	//---------------------------
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	int ismBytes = aBlockDim.x * sizeof(float) * 2;
	//---------------------------
	float* gfPeakLoc = 0L;
	cudaMalloc(&gfPeakLoc, sizeof(float));
	//---------------------------
	float fMask0 = 50.0f;
	int iPeak = 0;
	for(int i=1; i<20; i++)
	{	float fMask = fMask0 + i * 10.0f;
		mGFindPeak<<<aGridDim, aBlockDim, ismBytes>>>(m_gfRotSum,
	   	   m_aiCmpSize[0], iHalfY, fMask, gfPeakLoc);
		//---------------------------
		int iBytes = sizeof(float);
		float fPeakLoc = 0.0f;
		cudaMemcpy(&fPeakLoc, gfPeakLoc, iBytes, cudaMemcpyDefault);
		//-------------------
		iPeak = (int)(fPeakLoc + 0.5f);
		int x = iPeak % m_aiCmpSize[0];
		int y = iPeak / m_aiCmpSize[1];
		//-------------------
		float fDist = sqrtf(x * x + y * y);
		if(fabs(fDist - fMask) <= 50) continue;
		else break;
	}
	//---------------------------
	if(gfPeakLoc != 0L) cudaFree(gfPeakLoc);
	m_aiAvgPeak[0] = iPeak % m_aiCmpSize[0];
	m_aiAvgPeak[1] = iPeak / m_aiCmpSize[0];
}

void GFindTwinPeaks::mFindTwinPeaks(void)
{
	int iHalfY = m_aiCmpSize[1] / 2;
	float fRadius = 10.0f;
	int iCentX1 = m_aiAvgPeak[0];
	int iCentY1 = m_aiAvgPeak[1] + iHalfY; // gfHaflSpect at (0, iHafY)
	int aiPeak1[2] = {0};
	mLocalSearch(iCentX1, iCentY1, fRadius, aiPeak1);
	//---------------------------
	int iCentX2 = m_aiAvgPeak[1];
	int iCentY2 = -iCentX1 + iHalfY;
	int aiPeak2[2] = {0};
	mLocalSearch(iCentX2, iCentY2, fRadius, aiPeak2);
	//---------------------------
	float fR1 = aiPeak1[0] * aiPeak1[0] + aiPeak1[1] * aiPeak1[1];
	float fR2 = aiPeak2[0] * aiPeak2[0] + aiPeak2[1] * aiPeak2[1];
	fR1 = (float)sqrtf(fR1);
	fR2 = (float)sqrtf(fR2);
	m_afPeak1[0] = 1.0f / (fR1 * m_fPixSize);
	m_afPeak2[0] = 1.0f / (fR2 * m_fPixSize);
	//---------------------------
	m_afPeak1[1] = (float)atan(aiPeak1[1] / (aiPeak1[0] + 1e-30));
	m_afPeak2[1] = (float)atan(aiPeak2[1] / (aiPeak2[0] + 1e-30));	
	m_afPeak1[1] *= (180.0f / 3.14159265f);
	m_afPeak2[1] *= (180.0f / 3.14159265f);
}

void GFindTwinPeaks::mLocalSearch
(	int iCentX, 
	int iCentY, 
	float fRadius,
	int* piPeakLoc
)
{	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	int ismBytes = aBlockDim.x * sizeof(float) * 2;
	//---------------------------
	float* gfPeakLoc = 0L;
	cudaMalloc(&gfPeakLoc, sizeof(float));
	//---------------------------
	mGLocalSearch<<<aGridDim, aBlockDim, ismBytes>>>(m_gfHalfSpect,
	   m_aiCmpSize[0], m_aiCmpSize[1], iCentX, iCentY,
	   fRadius, gfPeakLoc);
	//---------------------------
	float fPeakLoc = 0;
	cudaMemcpy(&fPeakLoc, gfPeakLoc, sizeof(float),
	   cudaMemcpyDefault);
	//---------------------------
	int iPeakLoc = (int)(fPeakLoc + 0.5f);
	piPeakLoc[0] = iPeakLoc % m_aiCmpSize[0];
	piPeakLoc[1] = iPeakLoc / m_aiCmpSize[0] - m_aiCmpSize[1] / 2;
}

