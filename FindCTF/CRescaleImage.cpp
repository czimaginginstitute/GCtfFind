#include "CFindCTFInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

CRescaleImage::CRescaleImage(void)
{
	m_fRawPixSize = 1.0f;
	m_fPixSizeN = 1.0f;
	m_fBinning = 1;
	//---------------------------
	m_gfPadImgN = 0L;
	m_pForFFT = new CCufft2D;
	m_pInvFFT = new CCufft2D;
	//---------------------------
	memset(m_aiRawSize, 0, sizeof(m_aiRawSize));
	memset(m_aiNewSize, 0, sizeof(m_aiNewSize));
}

CRescaleImage::~CRescaleImage(void)
{
	if(m_pForFFT != 0L) delete m_pForFFT;
	if(m_pInvFFT != 0L) delete m_pInvFFT;
	if(m_gfPadImgN != 0L) cudaFree(m_gfPadImgN);
}

void CRescaleImage::Clean(void)
{
	if(m_gfPadImgN != 0L) cudaFree(m_gfPadImgN);
	m_gfPadImgN = 0L;
	m_pForFFT->DestroyPlan();
	m_pInvFFT->DestroyPlan();
}

float* CRescaleImage::GetScaledImg(void)
{
	return m_gfPadImgN;
};

void CRescaleImage::Setup(int* piRawSize, float fRawPixSize)
{
	this->Clean();
	//---------------------------
	memcpy(m_aiRawSize, piRawSize, sizeof(m_aiRawSize));
	m_fRawPixSize = fRawPixSize;
	//---------------------------
	m_fBinning = 1.2f / fRawPixSize;
	if(m_fBinning <= 1) m_fBinning = 1.0f;
	m_fPixSizeN = m_fRawPixSize * m_fBinning;
	//---------------------------
	m_aiNewSize[0] = (int)(m_aiRawSize[0] / m_fBinning + 0.5f);
	m_aiNewSize[1] = (int)(m_aiRawSize[1] / m_fBinning + 0.5f);
	m_aiNewSize[0] = m_aiNewSize[0] / 2 * 2;
	m_aiNewSize[1] = m_aiNewSize[1] / 2 * 2;
	//---------------------------
	m_aiPadSizeN[0] = (m_aiNewSize[0] / 2 + 1) * 2;
	m_aiPadSizeN[1] = m_aiNewSize[1];
	//---------------------------
	bool bPad = true, bCmp = true;
	if(m_fBinning > 1)
	{	m_pForFFT->CreateForwardPlan(m_aiRawSize, !bPad);
		m_pInvFFT->CreateInversePlan(m_aiNewSize, !bCmp);
	}
	//---------------------------
	int iBytes = sizeof(float) * m_aiPadSizeN[0] * m_aiPadSizeN[1];
	cudaMalloc(&m_gfPadImgN, iBytes);
}

void CRescaleImage::DoIt(float* pfImage)
{	
	if(m_fBinning == 1)
	{	int iBytes = sizeof(float) * m_aiRawSize[0];
		for(int y=0; y<m_aiNewSize[1]; y++)
		{	float* pfSrc = &pfImage[y * m_aiRawSize[0]];
			float* gfDst = &m_gfPadImgN[y * m_aiPadSizeN[0]];
			cudaMemcpy(gfDst, pfSrc, iBytes, cudaMemcpyDefault);
		}
		return;
	}
	//---------------------------
	cufftComplex* gCmpRaw = m_pForFFT->ForwardH2G(pfImage);
	cufftComplex* gCmpNew = (cufftComplex*)m_gfPadImgN;
	//---------------------------
	GFtResize2D gFtResize;
	bool bSum = true;
	int aiSizeIn[] = {m_aiRawSize[0] / 2 + 1, m_aiRawSize[1]};
	int aiSizeOt[] = {m_aiNewSize[0] / 2 + 1, m_aiNewSize[1]};
	gFtResize.DownSample(gCmpRaw, aiSizeIn, gCmpNew, aiSizeOt, !bSum);
	m_pInvFFT->Inverse(gCmpNew);
	//---------------------------
	if(gCmpRaw != 0L) cudaFree(gCmpRaw);
}

