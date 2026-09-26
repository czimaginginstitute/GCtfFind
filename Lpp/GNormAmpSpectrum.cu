#include "CLppInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static __global__ void mGApplyRamp
(	float* gfHalfSpect,
	int iCmpY
)
{       int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------
	float fX = blockIdx.x * 0.5f / (gridDim.x - 1);
	float fY = (y - 0.5f * iCmpY) / iCmpY;
	float fW = powf((fX * fX + fY * fY) + 1.0f, 2.0f);
	//---------------------------
	gfHalfSpect[i] *= fW;
}

static __global__ void mGNorm
(	float* gfHalfSpect,
	int iCmpY,
	float fMean,
	float fStd
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	//---------------------------
	int i = y * gridDim.x + blockIdx.x;
	float fVal = gfHalfSpect[i];
	//---------------------------
	fVal = (fVal - fMean) / fStd;
	fVal = fminf(fVal, 3.0f);
	fVal = fmaxf(fVal, -3.0f);
	//---------------------------
	gfHalfSpect[i] = fVal;
}

GNormAmpSpectrum::GNormAmpSpectrum(void)
{
	m_aiCmpSize[0] = 0;
	m_aiCmpSize[1] = 0;
}

GNormAmpSpectrum::~GNormAmpSpectrum(void)
{
	this->Clean();
}

void GNormAmpSpectrum::Clean(void)
{
}

void GNormAmpSpectrum::SetCmpSize(int* piCmpSize)
{
	if(m_aiCmpSize[0] != piCmpSize[0]) this->Clean();
	else if(m_aiCmpSize[1] != piCmpSize[1]) this->Clean();
	//---------------------------
	m_aiCmpSize[0] = piCmpSize[0];
	m_aiCmpSize[1] = piCmpSize[1];
}

//--------------------------------------------------------------------
// 1. The goal is to make the spectrum intensity more even.
// 2. Applying ramp filter and suppress the first Fourier components.
//--------------------------------------------------------------------
void GNormAmpSpectrum::DoIt(float* gfHalfSpect)
{
	mApplyRamp(gfHalfSpect);
	mNorm(gfHalfSpect);
}

void GNormAmpSpectrum::mApplyRamp(float* gfHalfSpect)
{       
	dim3 aBlockDim(1, 32);
	dim3 aGridDim(m_aiCmpSize[0], 1);
	aGridDim.y = (m_aiCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	mGApplyRamp<<<aGridDim, aBlockDim>>>(gfHalfSpect, m_aiCmpSize[1]);
}

void GNormAmpSpectrum::mNorm(float* gfHalfSpect)
{
	GCalcMeanStd calcMeanStd;
	float fStd = calcMeanStd.DoStd(gfHalfSpect, m_aiCmpSize, false);
	float fMean = calcMeanStd.m_fMean;
	if(fStd <= 0) return;
	//---------------------------
	dim3 aBlockDim(1, 32);
	dim3 aGridDim(m_aiCmpSize[0], 1);
        aGridDim.y = (m_aiCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//---------------------------
	mGNorm<<<aGridDim, aBlockDim>>>(gfHalfSpect,
	   m_aiCmpSize[1], fMean, fStd);
}


