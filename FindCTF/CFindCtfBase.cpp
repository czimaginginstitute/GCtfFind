#include "CFindCTFInc.h"
#include "../CMainInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

CFindCtfBase::CFindCtfBase(void)
{
	m_fDfMin = 0.0f;
	m_fDfMax = 0.0f;
	m_fAstAng = 0.0f;
	m_fExtPhase = 0.0f;
	m_fPhaseRange = 0.0f;
	m_fScore = 0.0f;
	mInitPointers();
	memset(m_aiImgSize, 0, sizeof(m_aiImgSize));
}

CFindCtfBase::~CFindCtfBase(void)
{
	this->Clean();
}

void CFindCtfBase::mInitPointers(void)
{
	m_pCtfTheory = 0L;
	m_pGenAvgSpect = 0L;
	m_gfFullSpect = 0L;
}

void CFindCtfBase::Clean(void)
{
	if(m_pCtfTheory != 0L) delete m_pCtfTheory;
	if(m_pGenAvgSpect != 0L) delete m_pGenAvgSpect;
	if(m_gfFullSpect != 0L) cudaFree(m_gfFullSpect);
	mInitPointers();
}

void CFindCtfBase::Setup1(CCTFTheory* pCtfTheory)
{
	this->Clean();
	//------------
	CInput* pInput = CInput::GetInstance();
	m_aiCmpSize[1] = pInput->m_iTileSize;
	m_aiCmpSize[0] = m_aiCmpSize[1] / 2 + 1;
	m_pCtfTheory = pCtfTheory->GetCopy();
	//-----------------------------------
        m_afResRange[0] = 20.0f * pInput->m_fPixelSize;
        m_afResRange[1] = 3.5f * pInput->m_fPixelSize;
	//--------------------------------------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gfFullSpect, sizeof(float) * iCmpSize * 4);
	m_gfRawSpect = m_gfFullSpect + iCmpSize * 2;
	m_gfCtfSpect = m_gfFullSpect + iCmpSize * 3;
	//------------------------------------------
	m_pGenAvgSpect = new CGenAvgSpectrum;
}

void CFindCtfBase::Setup2(int* piImgSize)
{
	if(m_aiImgSize[0] == piImgSize[0] && 
	   m_aiImgSize[1] == piImgSize[1]) return;
	//----------------------------------------
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_pGenAvgSpect->SetSizes(m_aiImgSize, m_aiCmpSize[1]);
}

void CFindCtfBase::SetPhase(float fInitPhase, float fPhaseRange)
{
	m_fExtPhase = fInitPhase;
	m_fPhaseRange = fPhaseRange;
}

void CFindCtfBase::SetHalfSpect(float* pfCtfSpect)
{
	int iBytes = sizeof(float) * m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMemcpy(m_gfCtfSpect, pfCtfSpect, iBytes, cudaMemcpyDefault);
}

float* CFindCtfBase::GetHalfSpect(bool bRaw, bool bToHost)
{
	float* gfSpect = bRaw ? m_gfRawSpect : m_gfCtfSpect;
	if(!bToHost) return gfSpect;
	//--------------------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	float* pfSpect = new float[iCmpSize];
	cudaMemcpy(pfSpect, gfSpect, sizeof(float) * iCmpSize,
	   cudaMemcpyDefault);
	return pfSpect;
}

void CFindCtfBase::GetSpectSize(int* piSize, bool bHalf)
{
	piSize[0] = m_aiCmpSize[0];
	piSize[1] = m_aiCmpSize[1];
	if(!bHalf) piSize[0] = (piSize[0] - 1) * 2;
}

void CFindCtfBase::GenHalfSpectrum(float* pfImage)
{
	CInput* pInput = CInput::GetInstance();
	bool bLogSpect = (pInput->m_iLogSpect != 0) ? true : false;	
	m_pGenAvgSpect->DoIt(pfImage, m_gfRawSpect, bLogSpect);
	mRemoveBackground();
}

float* CFindCtfBase::GenFullSpectrum(void)
{
	float fAstRad = m_fAstAng * 0.017453f;
        m_pCtfTheory->SetDefocus(m_fDfMin, m_fDfMax, fAstRad);
	m_pCtfTheory->SetExtPhase(m_fExtPhase, true);
	//-------------------------------------------
	CSpectrumImage spectrumImage;
	spectrumImage.DoIt(m_gfCtfSpect, m_gfRawSpect, m_aiCmpSize,
	   m_pCtfTheory, m_afResRange, m_gfFullSpect);
	//--------------------------------------------
	int iPixels = (m_aiCmpSize[0] - 1) * 2 * m_aiCmpSize[1];
	float* pfFullSpect = new float[iPixels];
	cudaMemcpy(pfFullSpect, m_gfFullSpect, iPixels * sizeof(float),
	   cudaMemcpyDefault);
	return pfFullSpect;
}

void CFindCtfBase::SaveSpectrum(char* pcMrcFile)
{
	/*
	if(pcMrcFile == 0L) return;
	if(m_pfFullSpect == 0L) return;
	//--------------------------------
	int aiSize[] = {0, m_aiCmpSize[1]};
	aiSize[0] = (m_aiCmpSize[0] - 1) * 2;
	bool bClean = true, bAsync = true;
	CAsyncSingleSave singleSave;
	singleSave.DoIt(pcMrcFile, m_pfFullSpect, aiSize,
	   1.0f, !bClean, !bAsync);
	*/
}

void CFindCtfBase::ShowResult(void)
{
	char acResult[256] = {'\0'};
	sprintf(acResult, "%9.2f  %9.2f  %6.2f  %6.2f  %8.5f\n",
	   m_fDfMin, m_fDfMax, m_fAstAng, m_fExtPhase, m_fScore);
	printf("%s\n", acResult);
}

void CFindCtfBase::mRemoveBackground(void)
{
	CInput* pInput = CInput::GetInstance();
	bool bLogSpect = (pInput->m_iLogSpect != 0) ? true: false;
	float fPixelSize = m_pCtfTheory->GetPixelSize();
	//float fMinRes = fPixelSize / m_afResRange[0];
	float fMinRes = 1.0f / 15.0f;
	GRmBackground2D rmBackground;
	rmBackground.DoIt(m_gfRawSpect, m_gfCtfSpect, bLogSpect,
	   m_aiCmpSize, fMinRes);
	//--------------------------------------------
	// do not threshold if the spectrum is flat.
	//--------------------------------------------
	GCalcMoment2D calcMoment2D;
	calcMoment2D.SetSize(m_aiCmpSize, false);
	float fMean = calcMoment2D.DoIt(m_gfCtfSpect, 1, true);
	float fStd = calcMoment2D.DoIt(m_gfCtfSpect, 2, true);
	fStd = fStd - fMean * fMean;
	if(fStd < 1.0f) return;
	//-----------------	
	fStd = (float)sqrtf(fStd);
	float fMin = fMean - 1.0f * fStd;
	float fMax = fMean + 1.0f * fStd;
	GThreshold2D threshold2D;
	//threshold2D.DoIt(m_gfCtfSpect, fMin, fMax, m_aiCmpSize, false);
	//-----------------
	mLowpass();
	/*	
	CSaveImages saveImages;
	saveImages.OpenFile("/home/shawn.zheng/Temp/TestRm.mrc");
	saveImages.Setup(m_aiCmpSize, 1);
	saveImages.DoIt(0, m_gfCtfSpect, true);
	*/
}

void CFindCtfBase::mLowpass(void)
{
	GCalcSpectrum calcSpectrum;
	bool bFullPadded = true;
	calcSpectrum.GenFullSpect(m_gfCtfSpect, m_aiCmpSize,
	   m_gfFullSpect, bFullPadded);
	//-----------------
	CuUtilFFT::GFFT2D aGFFT2D;
	int aiFFTSize[] = {(m_aiCmpSize[0] - 1) * 2, m_aiCmpSize[1]};
	aGFFT2D.CreatePlan(aiFFTSize, true);
	aGFFT2D.Forward(m_gfFullSpect, true);
	//-----------------
	GLowpass2D lowpass2D;
	cufftComplex* gCmpFullSpect = (cufftComplex*)m_gfFullSpect;
	lowpass2D.DoBFactor(gCmpFullSpect, gCmpFullSpect,
	   m_aiCmpSize, 10.0f);
	//-----------------
	aGFFT2D.CreatePlan(aiFFTSize, false);
	aGFFT2D.Inverse(gCmpFullSpect);
	//-----------------
	int iFullSizeX = m_aiCmpSize[0] * 2;
	int iHalfX = m_aiCmpSize[0] - 1;
	size_t tBytes = sizeof(float) * m_aiCmpSize[0];
	for(int y=0; y<m_aiCmpSize[1]; y++)
	{	float* gfSrc = m_gfFullSpect + y * iFullSizeX + iHalfX;
		float* gfDst = m_gfCtfSpect + y * m_aiCmpSize[0];
		cudaMemcpy(gfDst, gfSrc, tBytes, cudaMemcpyDefault);
        }
}

