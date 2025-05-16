#include "CMainInc.h"
#include "FindCTF/CFindCTFInc.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>

using namespace GCTFFind;

CFindSeriesCtfs* CFindSeriesCtfs::m_pInstance = 0L;

CFindSeriesCtfs* CFindSeriesCtfs::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CFindSeriesCtfs;
	return m_pInstance;
}

void CFindSeriesCtfs::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CFindSeriesCtfs::CFindSeriesCtfs(void)
{
	m_pvFindCtf2D = 0L;
	m_pvRescaleImg = 0L;
}

CFindSeriesCtfs::~CFindSeriesCtfs(void)
{
	this->Clean();
}

void CFindSeriesCtfs::Clean(void)
{
	if(m_pvFindCtf2D != 0L)
	{	delete (CFindCtf2D*)m_pvFindCtf2D;
		m_pvFindCtf2D = 0L;
	}
	if(m_pvRescaleImg != 0L)
	{	delete (CRescaleImage*)m_pvRescaleImg;
		m_pvRescaleImg = 0L;
	}
}

void CFindSeriesCtfs::DoIt(void)
{
	this->Clean();
	//---------------------------
	m_pvFindCtf2D = new CFindCtf2D;
	m_pvRescaleImg = new CRescaleImage;
	//---------------------------
	CInput* pInput = CInput::GetInstance();
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	CLoadImages* pLoadImages = CLoadImages::GetInstance();
	CAsyncSaveImages* pAsyncSaveImages = CAsyncSaveImages::GetInstance();
	CSaveCtfResults* pSaveCtfResults = CSaveCtfResults::GetInstance();
	//---------------------------
	bool bPop = true, bRaw = true, bToHost = true;
	m_iNumDone = 0;
	m_pRefPackage = 0L;
	m_iNumPackages = pInputFolder->GetNumPackages();
	//---------------------------
	Util_Time utilTime;
	float fWaitTime = 0.0f;
	char acLog[512] = {'\0'}, acBuf[32] = {'\0'};
	//---------------------------
	while(m_iNumDone < m_iNumPackages)
	{	m_pPackage = pLoadImages->GetPackage(bPop);
		if(m_pPackage == 0L)
		{	pLoadImages->WaitForExit(0.01f);
			fWaitTime += 0.01f;
			continue;
		}
		sprintf(acBuf, "Loading time: %.2f (s)\n", fWaitTime);
		strcpy(acLog, acBuf);
		//--------------------------
		utilTime.Measure();
		mRescaleImage(m_iNumDone);
		mSetupFindCtf();
		//-----------------------------------------------
		// In case different images have different sizes,
		// If the same, no extra oeverheader.
		//-----------------------------------------------
		mProcessPackage(m_iNumDone);
		fWaitTime = utilTime.GetElapsedSeconds();
		sprintf(acBuf, "Processing time: %.2f (s)\n", fWaitTime);
		strcat(acLog, acBuf);
		//--------------------------
		pAsyncSaveImages->SetPackage(m_pPackage);
		m_iNumDone += 1;
		//-------------------------------------------------
		// Save imcomplte results in case crash
		//-------------------------------------------------
		if(m_iNumDone % 21 != 0) continue;
		pSaveCtfResults->SaveCTF();
		pSaveCtfResults->SaveImod();
	}
	printf("%s", acLog);
	//----------------------------------
	this->Clean();
}

void CFindSeriesCtfs::mRescaleImage(int iPackage)
{
	CInput* pInput = CInput::GetInstance();
	CRescaleImage* pRescaleImg = (CRescaleImage*)m_pvRescaleImg;
	pRescaleImg->Setup(m_pPackage->m_aiImgSize, pInput->m_fPixSize);
	pRescaleImg->DoIt(m_pPackage->m_pfImage);
}

void CFindSeriesCtfs::mSetupFindCtf(void)
{
	CInput* pInput = CInput::GetInstance();
	CFindCtf2D* pFindCtf2D = (CFindCtf2D*)m_pvFindCtf2D;
	CRescaleImage* pRescaleImg = (CRescaleImage*)m_pvRescaleImg;
	//---------------------------
	CCTFTheory aInputCtf;
        float fExtPhase = pInput->m_afExtPhase[0] * 0.017453f;
        aInputCtf.Setup(pInput->m_fKv, pInput->m_fCs,
           pInput->m_fAmpContrast, pRescaleImg->m_fPixSizeN,
           100.0f, fExtPhase);
	//---------------------------
        pFindCtf2D->Setup1(&aInputCtf);
	pFindCtf2D->Setup2(pRescaleImg->m_aiNewSize);
	pFindCtf2D->GenHalfSpectrum(pRescaleImg->GetScaledImg());
}

void CFindSeriesCtfs::mProcessPackage(int iPackage)
{
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	CFindCtf2D* pFindCtf2D = (CFindCtf2D*)m_pvFindCtf2D;
	//---------------------------
	/*if(iPackage == 0 || !pInputFolder->IsTomo()) mProcessFull();
	if(iPackage == 0) 
	{	mProcessFull();
	}
	else 
	{	mProcessRefine();
		if(m_pPackage->m_fScore < 0.1) mProcessFull();
	}*/
	mProcessFull();
	//--------------------
	mDisplay();
	//---------------------------
	bool bHalf = true;
	m_pPackage->m_pfFullSpect = pFindCtf2D->GenFullSpectrum();
	pFindCtf2D->GetSpectSize(m_pPackage->m_aiSpectSize, !bHalf);
}

void CFindSeriesCtfs::mProcessFull(void)
{
	CFindCtf2D* pFindCtf2D = (CFindCtf2D*)m_pvFindCtf2D;
	CInput* pInput = CInput::GetInstance();
	//-------------------------------------
        float fPhaseRange = fmaxf(pInput->m_afExtPhase[1], 0.0f);
	pFindCtf2D->SetPhase(pInput->m_afExtPhase[0], fPhaseRange);
	//---------------------------------------------------------
	pFindCtf2D->Do2D();
	mGetResults();
	//------------
	m_pRefPackage = m_pPackage;
}

void CFindSeriesCtfs::mProcessRefine(void)
{
	CInput* pInput = CInput::GetInstance();
	CInputFolder* pInputFolder = CInputFolder::GetInstance();	
	CFindCtf2D* pFindCtf2D = (CFindCtf2D*)m_pvFindCtf2D;
	bool bTomo = pInputFolder->IsTomo();
	//--------------------------------------------------
	float afDfRange[2] = {0.0f}, afAstRatio[2] = {0.0f};
	float afAstAngle[2] = {0.0f}, afExtPhase[2] = {0.0f};
	//---------------------------------------------------
	float fDfMin = m_pRefPackage->m_fDfMin;
	float fDfMax = m_pRefPackage->m_fDfMax;
	float fDfRange = 5000.0f * pInput->m_fPixSize 
	   * pInput->m_fPixSize;
	afDfRange[0] = 0.5f * (fDfMin + fDfMax); 
	afDfRange[1] = fmaxf(afDfRange[0] * 0.5f, fDfRange);
	afAstRatio[0] = CFindCtfHelp::CalcAstRatio(fDfMin, fDfMax);
	afAstAngle[0] = m_pRefPackage->m_fAzimuth;
	afExtPhase[0] = m_pRefPackage->m_fExtPhase;
	//-----------------------------------------
	if(!bTomo)
	{	afAstRatio[1] = 0.04f;
		afAstAngle[1] = 30.0f;
		if(pInput->m_afExtPhase[1] > 0) 
		{	afExtPhase[1] = 40.0f;
		}
	}
	//------------------------------------------------------------
	pFindCtf2D->Refine(afDfRange, afAstRatio, 
	   afAstAngle, afExtPhase);
	mGetResults();
	//-------------------------
	if(m_pRefPackage->m_fScore < m_pPackage->m_fScore)
	{	m_pRefPackage = m_pPackage;
	}
}

void CFindSeriesCtfs::mGetResults(void)
{
	CFindCtf2D* pFindCtf2D = (CFindCtf2D*)m_pvFindCtf2D;
	m_pPackage->m_fDfMin = pFindCtf2D->m_fDfMin;
	m_pPackage->m_fDfMax = pFindCtf2D->m_fDfMax;
	m_pPackage->m_fAzimuth = pFindCtf2D->m_fAstAng;
	m_pPackage->m_fExtPhase = pFindCtf2D->m_fExtPhase;
	m_pPackage->m_fScore = pFindCtf2D->m_fScore;
	m_pPackage->m_fCtfRes = pFindCtf2D->m_fCtfRes;
}

void CFindSeriesCtfs::mDisplay(void)
{
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	//-------------------------------------------------------
	char acMainFile[256] = {'\0'};
        pInputFolder->GetFileName(m_pPackage->m_iImgIdx, acMainFile);
	//-----------------------------------------------------------
	char acInfo[512] = {'\0'};
        sprintf(acInfo, "%s: %5d prcoessed, %5d left\n", acMainFile,
	   m_iNumDone + 1, m_iNumPackages - 1 - m_iNumDone);
	//--------------------------------------------------
	char acBuf1[128] = {'\0'};
	sprintf(acBuf1, "%s", "   Index  dfmin     dfmax    "
	   "azimuth  phase   Res(A)  score\n");
	strcat(acInfo, acBuf1);
	//---------------------
	char acBuf2[128] = {'\0'};
	sprintf(acBuf2, "   %4d  %8.2f  %8.2f  %6.2f %6.2f  %6.2f %9.5f\n",
	   m_pPackage->m_iImgIdx + 1, m_pPackage->m_fDfMin,
	   m_pPackage->m_fDfMax,      m_pPackage->m_fAzimuth,
	   m_pPackage->m_fExtPhase,   m_pPackage->m_fCtfRes,
	   m_pPackage->m_fScore);
	strcat(acInfo, acBuf2);
	//---------------------
	printf("%s\n", acInfo);
}
