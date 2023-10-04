#include "CFindCTFInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

//-----------------------------------------------------------------------------
// 1. Calculate the logrithmic amplitude spectrum given gComp, the Fourier
//    transform.
// 2. The spectrum is the half spectrum with x frequency ranging from 0 to
//    0.5 and y frequency form -0.5 to +0.5.
// 3. The zero frequency is at (0, iCmpY / 2).
//-----------------------------------------------------------------------------
static __global__ void mGCalcSpectrum
(	cufftComplex* gComp, 
	float* gfSpectrum,
	int iType, // 1 - power, 2 - amp, 3 - log
	int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	y = y + iCmpY / 2;
	if(y >= iCmpY) y = y - iCmpY;
	//---------------------------
	float fVal = gComp[i].x * gComp[i].x + gComp[i].y * gComp[i].y;
	if(iType == 2) fVal = sqrtf(fVal);
	else if(iType == 2) fVal = logf(fVal + 1.0f);
	gfSpectrum[y * gridDim.x + blockIdx.x] = fVal;
}

static __global__ void mGCenterSpectrum
(	float* gfSpectrum,
	int iHalfY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iHalfY) return;
	int i1 = y * gridDim.x + blockIdx.x;
	int i2 = (y + iHalfY) * gridDim.x + blockIdx.x;
	float f1 = gfSpectrum[i1];
	gfSpectrum[i1] = gfSpectrum[i2];
	gfSpectrum[i2] = f1;
}

GCalcSpectrum2D::GCalcSpectrum2D(void)
{
	m_pvGFFT2D = new CuUtilFFT::GFFT2D;
}

GCalcSpectrum2D::~GCalcSpectrum2D(void)
{
	if(m_pvGFFT2D == 0L) return;
	delete (CuUtilFFT::GFFT2D*)m_pvGFFT2D;
}

void GCalcSpectrum2D::Setup(int* piImgSize, bool bPadded)
{
	CuUtilFFT::GFFT2D* pGFFT2D = (CuUtilFFT::GFFT2D*)m_pvGFFT2D;
	pGFFT2D->DestroyPlan();
	int aiFFTSize[2] = {piImgSize[0], piImgSize[1]};
	if(bPadded) aiFFTSize[0] = (piImgSize[0] / 2 - 1) * 2;
	pGFFT2D->CreatePlan(aiFFTSize, true); // forward plan
	//------------------------------------
	m_aiCmpSize[0] = aiFFTSize[0] / 2 + 1;
	m_aiCmpSize[1] = aiFFTSize[1];
}

void GCalcSpectrum2D::DoPow
(	float* gfPadImg,
	float* gfSpectrum
)
{	CuUtilFFT::GFFT2D* pGFFT2D = (CuUtilFFT::GFFT2D*)m_pvGFFT2D;
	pGFFT2D->Forward(gfPadImg, true); // normalized
	cufftComplex* gCmpImg = (cufftComplex*)gfPadImg;
	mDoIt(gCmpImg, gfSpectrum, 1);
}

void GCalcSpectrum2D::DoAmp
(	float* gfPadImg,
	float* gfSpectrum
)
{	CuUtilFFT::GFFT2D* pGFFT2D = (CuUtilFFT::GFFT2D*)m_pvGFFT2D;
	pGFFT2D->Forward(gfPadImg, true);
	cufftComplex* gCmpImg = (cufftComplex*)gfPadImg;
	mDoIt(gCmpImg, gfSpectrum, 2);
}

void GCalcSpectrum2D::DoLog
(	float* gfPadImg,
	float* gfSpectrum
)
{	CuUtilFFT::GFFT2D* pGFFT2D = (CuUtilFFT::GFFT2D*)m_pvGFFT2D;
	pGFFT2D->Forward(gfPadImg, true);
	cufftComplex* gCmpImg = (cufftComplex*)gfPadImg;
	mDoIt(gCmpImg, gfSpectrum, 3);
}	

void GCalcSpectrum2D::mDoIt
(	cufftComplex* gCmpImg, 
	float* gfSpectrum, 
	int iType
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiCmpSize[0], 1);
	aGridDim.y = (m_aiCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	mGCalcSpectrum<<<aGridDim, aBlockDim>>>(gCmpImg,
	   gfSpectrum, m_aiCmpSize[1], iType);
}

void GCalcSpectrum2D::mCenterSpectrum(float* gfSpectrum)
{
	int iHalfY = m_aiCmpSize[0] / 2;
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiCmpSize[0], 1);
	aGridDim.y = (m_aiCmpSize[1] + aBlockDim.y - 1) / iHalfY;
	mGCenterSpectrum<<<aGridDim, aBlockDim>>>(gfSpectrum, iHalfY);
}
