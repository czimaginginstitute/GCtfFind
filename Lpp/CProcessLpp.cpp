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

CProcessLpp* CProcessLpp::m_pInstance = 0L;

CProcessLpp* CProcessLpp::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CProcessLpp;
	return m_pInstance;
}

void CProcessLpp::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CProcessLpp::CProcessLpp(void)
{
	m_gfBuf = 0L;
	m_ppfAmps = 0L;
	m_ppfPhis = 0L;
	m_iNumAmps = 0;
}

CProcessLpp::~CProcessLpp(void)
{
	this->Clean();
	this->mCleanAmps();
}

void CProcessLpp::Clean(void)
{
	if(m_gfBuf != 0L)
	{	cudaFree(m_gfBuf);
		m_gfBuf = 0L;
	}
	//---------------------------
	int iSize = m_ampQueue.size();
	for(int i=0; i<iSize; i++)
	{	float* pfImg = m_ampQueue.front();
		if(pfImg != 0L) delete[] pfImg;
		m_ampQueue.pop();
	}
	//---------------------------
	iSize = m_phiQueue.size();
	for(int i=0; i<iSize; i++)
	{	float* pfImg = m_phiQueue.front();
		if(pfImg != 0L) delete[] pfImg;
		m_phiQueue.pop();
	}
}

void CProcessLpp::Setup(int* piImgSize, float fBinning)
{
	this->Clean();
	CPad2D pad2D;
	//---------------------------
	m_aiRawSize[0] = piImgSize[0];
	m_aiRawSize[1] = piImgSize[1];
	//---------------------------
	int iSize = piImgSize[0];
	if(piImgSize[1] < piImgSize[0]) iSize = piImgSize[1];
	m_aiImgSize[0] = iSize;
	m_aiImgSize[1] = iSize;
	pad2D.GetPadSize(m_aiImgSize, m_aiImgPadSize);
	//---------------------------
	int iBinSize = (int)(m_aiImgSize[0] / fBinning) / 2 * 2;
        m_aiBinSize[0] = iBinSize;
        m_aiBinSize[1] = iBinSize;
        pad2D.GetPadSize(m_aiBinSize, m_aiBinPadSize);
	//---------------------------
	int iImgPadSize = m_aiImgPadSize[0] * m_aiImgPadSize[1];
	int iBinPadSize = m_aiBinPadSize[0] * m_aiBinPadSize[1];
	int iBufSize = 2 * iBinPadSize;
	if(iBufSize < iImgPadSize) iBufSize = iImgPadSize;
	m_gfBuf = CSimpleFuncs::GAllocFloat(iBufSize);
}

void CProcessLpp::DoIt(float* pfImage)
{
	mPadImage(pfImage);	
	mNormImg(m_gfBuf, m_aiImgPadSize, true);
	mRoundEdge(m_gfBuf, m_aiImgPadSize, true);
	//---------------------------
	mForwardFFT();
	cufftComplex* gBinCmp = mFtBinning();
	mCalcAmpPhase(gBinCmp);
	if(gBinCmp != 0L) cudaFree(gBinCmp);
	mNormImg(m_gfBuf, m_aiBinPadSize, true);
	//---------------------------
	int iBinPadSize = m_aiBinPadSize[0] * m_aiBinPadSize[1];
	float* gfPhase = &m_gfBuf[iBinPadSize];
	//---------------------------
	bool bAmp = true;
	mQueueImg(m_gfBuf, m_aiBinPadSize, bAmp);
	mQueueImg(gfPhase, m_aiBinPadSize, !bAmp);
}

void CProcessLpp::PostProcess(void)
{
	mCleanAmps();
	m_iNumAmps = m_ampQueue.size();
	if(m_iNumAmps == 0) return;
	//---------------------------
	m_ppfAmps = new float*[m_iNumAmps];
	m_ppfPhis = new float*[m_iNumAmps];
	for(int i=0; i<m_iNumAmps; i++)
	{	m_ppfAmps[i] = m_ampQueue.front();
		m_ampQueue.pop();
		//-------------------
		m_ppfPhis[i] = m_phiQueue.front();
		m_phiQueue.pop();	
	}
	//---------------------------
	mCorrelateAmps();
	//---------------------------
	mSaveStack(m_ppfAmps, m_aiBinSize, m_iNumAmps, "_Amp.mrc");
	mSaveStack(m_ppfPhis, m_aiBinSize, m_iNumAmps, "_Phi.mrc");
	mCleanAmps();
}

void CProcessLpp::mPadImage(float* pfImage)
{
	int iOffsetX = (m_aiRawSize[0] - m_aiImgSize[0]) / 2;
	int iOffsetY = (m_aiRawSize[1] - m_aiImgSize[1]) / 2;
	int iBytes = sizeof(float) * m_aiImgSize[0];
	//---------------------------
	for(int y=0; y<m_aiImgSize[1]; y++)
	{	int ySrc = y + iOffsetY;
		float* pfSrc = &pfImage[ySrc * m_aiRawSize[0] + iOffsetX];
		float* gfDst = &m_gfBuf[y * m_aiImgPadSize[0]];
		cudaMemcpy(gfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

void CProcessLpp::mNormImg(float* gfImg, int* piImgSize, bool bPadded)
{
	GCalcMeanStd calcMeanStd;
	float fStd = calcMeanStd.DoStd(gfImg, piImgSize, bPadded);
	float fMean = calcMeanStd.m_fMean;
	//---------------------------
	GNormalize2D norm2D;
	norm2D.DoIt(gfImg, fMean, fStd, piImgSize, bPadded);
}

void CProcessLpp::mRoundEdge(float* gfImg, int* piImgSize, bool bPadded)
{
	GRoundEdge aGRoundEdge;
	int iImgX = piImgSize[0];
	if(bPadded) iImgX = (iImgX / 2 - 1) * 2;
        float afCent[] = {iImgX * 0.5f, piImgSize[1] * 0.5f};
	float afSize[] = {(float)iImgX, (float)piImgSize[1]};
	aGRoundEdge.SetMask(afCent, afSize);
	aGRoundEdge.DoIt(m_gfBuf, piImgSize);
}

void CProcessLpp::mForwardFFT(void)
{
	CCufft2D cufft2D;
	cufft2D.CreateForwardPlan(m_aiImgSize, false);
	cufft2D.Forward(m_gfBuf);
	cudaStreamSynchronize((cudaStream_t)0);
}

cufftComplex* CProcessLpp::mFtBinning(void)
{
	int aiInCmpSize[2], aiOutCmpSize[2];
	CPad2D pad2D;
	pad2D.GetCmpSize(m_aiImgSize, aiInCmpSize);
	pad2D.GetCmpSize(m_aiBinSize, aiOutCmpSize);
	//---------------------------
	cufftComplex* gInCmp = (cufftComplex*)m_gfBuf; 
	cufftComplex* gOutCmp = CSimpleFuncs::GAllocCmp(aiOutCmpSize);
	//---------------------------
	GFtResize2D ftResize;
	ftResize.DownSample(gInCmp, aiInCmpSize,
	   gOutCmp, aiOutCmpSize, false);
	return gOutCmp;
}

void CProcessLpp::mCalcAmpPhase(cufftComplex* gCmp)
{
	CPad2D pad2D;
	GCalcSpectrum calcSpect;
	GCalcPhase2D calcPhase;
	//---------------------------
	int aiCmpSize[2] = {0};
	pad2D.GetCmpSize(m_aiBinSize, aiCmpSize);
	float* gfHalfBuf = CSimpleFuncs::GAllocFloat(aiCmpSize);
	//---------------------------
	calcSpect.DoIt(gCmp, gfHalfBuf, aiCmpSize);
	calcSpect.ApplyRamp(gfHalfBuf, aiCmpSize);
	calcSpect.GenFullSpect(gfHalfBuf, aiCmpSize,
	   m_gfBuf, true);
	//---------------------------
	int iBinPadSize = m_aiBinPadSize[0] * m_aiBinPadSize[1];
	float* gfFullPhase = &m_gfBuf[iBinPadSize];
	calcPhase.DoHalf(gCmp, gfHalfBuf, aiCmpSize);
	calcPhase.DoFull(gfHalfBuf, aiCmpSize,
	   gfFullPhase, true);
	//---------------------------
	if(gfHalfBuf != 0L) cudaFree(gfHalfBuf);
}

void CProcessLpp::mQueueImg(float* gfPadImg, int* piPadSize, bool bAmp)
{
	CPad2D pad2D;
	int aiImgSize[2] = {0};
	pad2D.GetImgSize(piPadSize, aiImgSize);
	//---------------------------
	int iImgSize = aiImgSize[0] * aiImgSize[1];
	float* pfImg = new float[iImgSize];
	pad2D.Unpad(gfPadImg, piPadSize, pfImg);
	if(bAmp) m_ampQueue.push(pfImg);
	else m_phiQueue.push(pfImg);
}

void CProcessLpp::mCorrelateAmps(void)
{
	if(m_iNumAmps <= 1) return;
	int iSize = m_iNumAmps * m_iNumAmps;
	float* pfAmpCCs = new float[iSize];
	//---------------------------
	float *gfAmp1 = 0L, *gfAmp2 = 0L;
	int iBytes = m_aiBinSize[0] * m_aiBinSize[1] * sizeof(float);
	cudaMalloc(&gfAmp1, iBytes);
	cudaMalloc(&gfAmp2, iBytes);
	//---------------------------
	GCalcCC2D calcCC2D;
	for(int y=0; y<m_iNumAmps; y++)
	{	cudaMemcpy(gfAmp1, m_ppfAmps[y], iBytes, 
		   cudaMemcpyDefault);
		for(int x=0; x<m_iNumAmps; x++)
		{	int k = y * m_iNumAmps + x;
			if(x <= y) continue;
			cudaMemcpy(gfAmp2, m_ppfAmps[x], iBytes, 
			   cudaMemcpyDefault);
			pfAmpCCs[k] = calcCC2D.DoIt(gfAmp1, gfAmp2,
			   m_aiBinSize, false);
		}
	}
	//---------------------------
	for(int i=0; i<iSize; i++)
	{	int x = i % m_iNumAmps;
		int y = i / m_iNumAmps;
		int j = x * m_iNumAmps + y;
		//-------------------
		if(x > y) continue;
		else if(x == y) pfAmpCCs[i] = 1.0f;
		else pfAmpCCs[i] = pfAmpCCs[j];
	}
	//---------------------------
	if(gfAmp1 != 0L) cudaFree(gfAmp1);
	if(gfAmp2 != 0L) cudaFree(gfAmp2);
	//---------------------------
	CInput* pInput = CInput::GetInstance();
	char acCCFile[512] = {'\0'};
	pInput->GetOutFile("_Amp_CC.txt", acCCFile);
	FILE* pFile = fopen(acCCFile, "w");
	//---------------------------	
	char acLine[512] = {'\0'};
	char acBuf[16] = {'\0'};
	for(int y=0; y<m_iNumAmps; y++)
	{	memset(acLine, 0, sizeof(acLine));
		for(int x=0; x<m_iNumAmps; x++)
		{	int i = y * m_iNumAmps + x;
			sprintf(acBuf, "%10.4f", pfAmpCCs[i]);
			if(x == 0) strcpy(acLine, acBuf);
			else strcat(acLine, acBuf);
		}
		fprintf(pFile, "%s\n", acLine);
	}
	if(pFile != 0L) fclose(pFile);
	if(pfAmpCCs != 0L) delete[] pfAmpCCs;
}

void CProcessLpp::mCleanAmps(void)
{
	if(m_iNumAmps == 0) return;
	for(int i=0; i<m_iNumAmps; i++)
	{	if(m_ppfAmps[i] != 0L) delete[] m_ppfAmps[i];
		if(m_ppfPhis[i] != 0L) delete[] m_ppfPhis[i];
	}
	delete[] m_ppfAmps;
	delete[] m_ppfPhis;
	m_iNumAmps = 0;
	m_ppfAmps = 0L;
	m_ppfPhis = 0L;
}

void CProcessLpp::mSaveImg
(	float* gfPadImg, 
	int* piPadSize, 
	const char* pcSuffix
)
{	CInput* pInput = CInput::GetInstance();
	char acOutMrc[256] = {'\0'};
	pInput->GetOutFile(pcSuffix, acOutMrc);
	//---------------------------
	CPad2D pad2D;
	int aiImgSize[2] = {0};
	pad2D.GetImgSize(piPadSize, aiImgSize);
	float* pfImg = new float[aiImgSize[0] * aiImgSize[1]];
	pad2D.Unpad(gfPadImg, piPadSize, pfImg);
	//---------------------------
	CSaveTempMrc saveMrc;
	saveMrc.SetFile(acOutMrc, "");
	saveMrc.DoIt(pfImg, 2, aiImgSize);
	if(pfImg != 0L) delete[] pfImg;
}

void CProcessLpp::mSaveStack
(       float** ppfImgs,
        int* piImgSize,
	int iNumImgs,
        const char* pcSuffix
)
{       CInput* pInput = CInput::GetInstance();
        char acOutMrc[256] = {'\0'};
        pInput->GetOutFile(pcSuffix, acOutMrc);
        //---------------------------
        CSaveTempMrc saveMrc;
        saveMrc.SetFile(acOutMrc, "");
	saveMrc.DoStack(ppfImgs, piImgSize, iNumImgs);
}

