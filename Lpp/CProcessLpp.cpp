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
	m_pCalcAmpSpect = new CCalcAmpSpect;
	m_pFindTwinPeaks = new GFindTwinPeaks;
}

CProcessLpp::~CProcessLpp(void)
{
	if(m_pCalcAmpSpect != 0L) delete m_pCalcAmpSpect;
	if(m_pFindTwinPeaks != 0L) delete m_pFindTwinPeaks;
	this->Clean();
}

void CProcessLpp::Clean(void)
{
}

void CProcessLpp::Setup(int* piImgSize, float fPixSize)
{
	m_fPixSize = fPixSize;
	m_pCalcAmpSpect->Setup(piImgSize, fPixSize);
	m_pFindTwinPeaks->SetPadSize(m_pCalcAmpSpect->m_aiPadSize);
	m_pFindTwinPeaks->SetPixSize(m_fPixSize);
}

void CProcessLpp::DoIt(void* pvCtfPackage)
{
	m_pvCtfPackage = pvCtfPackage;
	CCtfPackage* pCtfPackage = (CCtfPackage*)pvCtfPackage;
	//---------------------------
	m_pCalcAmpSpect->DoIt(pCtfPackage->m_pfImage);
	float* gfPadSpect = m_pCalcAmpSpect->m_gfPadSpect;
	int* piPadSize = m_pCalcAmpSpect->m_aiPadSize;
	//---------------------------
	mLowpass();
	mSaveFullSpect();	
	//---------------------------
	m_pFindTwinPeaks->DoIt(m_pCalcAmpSpect->m_gfPadSpect);
	int iBytes = sizeof(float) * 2;
	memcpy(pCtfPackage->m_afLpp1, m_pFindTwinPeaks->m_afPeak1, iBytes);
	memcpy(pCtfPackage->m_afLpp2, m_pFindTwinPeaks->m_afPeak2, iBytes);
}

void CProcessLpp::mLowpass(void)
{
	float* gfPadSpect = m_pCalcAmpSpect->m_gfPadSpect;
	int* piPadSize = m_pCalcAmpSpect->m_aiPadSize;
	//---------------------------
	float fCutoff = 30.0f;
	float fShell = (piPadSize[1] * m_fPixSize) / fCutoff;
	//---------------------------
	GRoundEdge2D roundEdge; // util
	roundEdge.SetMask(fShell, fShell);
	roundEdge.DoIt(gfPadSpect, piPadSize, true, 200.0f);
}

void CProcessLpp::mSaveFullSpect(void)
{
	float* gfPadSpect = m_pCalcAmpSpect->m_gfPadSpect;
	int* piPadSize = m_pCalcAmpSpect->m_aiPadSize;
	//---------------------------	
	char acMrcName[256] = {'\0'};
	CCtfPackage* pCtfPackage = (CCtfPackage*)m_pvCtfPackage;
	char* pcInMrc = pCtfPackage->m_acMrcFileName;
	char* pcSlash = strrchr(pcInMrc, '/');
	if(pcSlash == 0L) strcpy(acMrcName, pcInMrc);
	else strcpy(acMrcName, &pcSlash[1]);
	//---------------------------
	CInput* pInput = CInput::GetInstance();
	char acOutMrc[512] = {'\0'};
	pInput->GetOutFile(acMrcName, "_AMP.mrc", acOutMrc);
	//---------------------------
	/*
	CPad2D pad2D;
	int aiImgSize[2] = {0};
	pad2D.GetImgSize(piPadSize, aiImgSize);
	float* pfImg = new float[aiImgSize[0] * aiImgSize[1]];
	pad2D.Unpad(gfPadSpect, piPadSize, pfImg);
	*/
	int aiImgSize[2] = {0};
	aiImgSize[0] = (int)(piPadSize[1] * m_fPixSize / 10.0f) / 2 * 2;
	aiImgSize[1] = aiImgSize[0];
	//---------------------------
	float* pfImg = new float[aiImgSize[0] * aiImgSize[1]];
	int iOffset = (piPadSize[1] - aiImgSize[1]) / 2;
	int iBytes = sizeof(float) * aiImgSize[0];
	//---------------------------
	int iOffsetSrc = iOffset * piPadSize[0] + iOffset;
	for(int y=0; y<aiImgSize[1]; y++)
	{	float* gfSrc = &gfPadSpect[y * piPadSize[0] + iOffsetSrc];
		float* pfDst = &pfImg[y * aiImgSize[0]];
		cudaMemcpy(pfDst, gfSrc, iBytes, cudaMemcpyDefault);
	}
	//---------------------------
	CSaveTempMrc saveMrc;
	char acExt[16] = {'\0'};
	saveMrc.SetFile(acOutMrc, acExt);
	saveMrc.DoIt(pfImg, 2, aiImgSize);
	if(pfImg != 0L) delete[] pfImg;
}

