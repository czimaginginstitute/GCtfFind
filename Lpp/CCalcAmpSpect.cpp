#include "CLppInc.h"
#include "../CMainInc.h"
#include "../FindCTF/CFindCTFInc.h"
#include "../Util/CUtilInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static void mGSaveTempMrc
(	float* gfImg, 
	int* piSize,
	const char* pcMrcName
)
{	CInput* pInput = CInput::GetInstance();
        char acOutMrc[256] = {'\0'};
        pInput->GetOutFile(pcMrcName, 0L, acOutMrc);
        //---------------------------
        CSaveTempMrc saveMrc;
	char acExt[16] = {'\0'};
        saveMrc.SetFile(acOutMrc, acExt);
        saveMrc.GDoIt(gfImg, piSize);
}	

CCalcAmpSpect::CCalcAmpSpect(void)
{
	m_gfPadImg = 0L;
	m_pCufft2D = 0L;
	m_aiImgSize[0] = 0;
	m_aiImgSize[1] = 0;
}

CCalcAmpSpect::~CCalcAmpSpect(void)
{
	this->Clean();
}

void CCalcAmpSpect::Clean(void)
{
	if(m_gfPadImg != 0L)
	{	cudaFree(m_gfPadImg);
		m_gfPadImg = 0L;
	}
	if(m_pCufft2D != 0L)
	{	delete m_pCufft2D;
		m_pCufft2D = 0L;
	}
}

void CCalcAmpSpect::Setup(int* piImgSize, float fPixSize)
{
	if(m_aiRawSize[0] != piImgSize[0]) this->Clean();
	else if(m_aiRawSize[1] != piImgSize[1]) this->Clean();
	//---------------------------
	m_aiRawSize[0] = piImgSize[0];
	m_aiRawSize[1] = piImgSize[1];
	m_fPixSize = fPixSize;
	//---------------------------
	if(m_gfPadImg != 0L) return;
	//---------------------------
	// crop into square image
	//---------------------------
	int iSize = piImgSize[0];
	if(piImgSize[1] < piImgSize[0]) iSize = piImgSize[1];
	m_aiImgSize[0] = iSize;
	m_aiImgSize[1] = iSize;
	//---------------------------
	m_aiPadSize[0] = (m_aiImgSize[0] / 2 + 1) * 2;
	m_aiPadSize[1] = m_aiImgSize[1];
	m_aiCmpSize[0] = m_aiPadSize[0] / 2;
	m_aiCmpSize[1] = m_aiPadSize[1];
	//---------------------------
	m_gfPadImg = CSimpleFuncs::GAllocFloat(m_aiPadSize);
	m_gfPadSpect = CSimpleFuncs::GAllocFloat(m_aiPadSize);
	//---------------------------
	m_pCufft2D = new CCufft2D;
	bool bPadded = true;
	m_pCufft2D->CreateForwardPlan(m_aiImgSize, !bPadded);
}

void CCalcAmpSpect::DoIt(float* pfImage)
{
	mPadImage(pfImage);	
	mNormImg();
	mRoundEdge();
	mForwardFFT();
	mCalcSpect();
}

void CCalcAmpSpect::mPadImage(float* pfImage)
{
	int iPadSize = m_aiPadSize[0] * m_aiPadSize[1];
	cudaMemset(m_gfPadImg, 0, sizeof(float) * iPadSize);

	int iOffsetX = (m_aiRawSize[0] - m_aiImgSize[0]) / 2;
	int iOffsetY = (m_aiRawSize[1] - m_aiImgSize[1]) / 2;
	int iBytes = sizeof(float) * m_aiImgSize[0];
	//---------------------------
	for(int y=0; y<m_aiImgSize[1]; y++)
	{	int ySrc = y + iOffsetY;
		float* pfSrc = &pfImage[ySrc * m_aiRawSize[0] + iOffsetX];
		float* gfDst = &m_gfPadImg[y * m_aiPadSize[0]];
		cudaMemcpy(gfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

void CCalcAmpSpect::mNormImg(void)
{
	bool bPadded = true;
	GCalcMeanStd calcMeanStd;
	float fStd = calcMeanStd.DoStd(m_gfPadImg, m_aiPadSize, bPadded);
	float fMean = calcMeanStd.m_fMean;
	//---------------------------
	GNormalize2D norm2D;
	norm2D.DoIt(m_gfPadImg, fMean, fStd, m_aiPadSize, bPadded);
}

void CCalcAmpSpect::mRoundEdge(void)
{
        float afCent[] = {m_aiImgSize[0] * 0.5f, m_aiImgSize[1] * 0.5f};
	float afSize[] = {(float)m_aiImgSize[0], (float)m_aiImgSize[1]};
	//---------------------------
	GRoundEdge aGRoundEdge;
	aGRoundEdge.SetMask(afCent, afSize);
	aGRoundEdge.DoIt(m_gfPadImg, m_aiPadSize);
}

void CCalcAmpSpect::mForwardFFT(void)
{
	m_pCufft2D->Forward(m_gfPadImg);
	cudaStreamSynchronize((cudaStream_t)0);
}

//--------------------------------------------------------------------
// 1. Half spectrum with DC centered at (0, Ny/2)
//--------------------------------------------------------------------
void CCalcAmpSpect::mCalcSpect(void)
{
	GCalcSpectrum calcSpect;
	float* gfHalfSpect = CSimpleFuncs::GAllocFloat(m_aiCmpSize);
	cufftComplex* gCmpImg = (cufftComplex*)m_gfPadImg;
	calcSpect.DoIt(gCmpImg, gfHalfSpect, m_aiCmpSize);
	//calcSpect.Logrithm(gfHalfSpect, m_aiCmpSize);
	//calcSpect.ApplyRamp(gfHalfSpect, m_aiCmpSize);
	//---------------------------
	bool bPadded = true;
	calcSpect.GenFullSpect(gfHalfSpect, m_aiCmpSize,
	   m_gfPadSpect, bPadded);
	//---------------------------
	if(gfHalfSpect != 0L) cudaFree(gfHalfSpect);
}

