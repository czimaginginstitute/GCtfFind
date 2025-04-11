#include "CMainInc.h"
#include "FindCTF/CFindCTFInc.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
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
}

void CFindSeriesCtfs::DoIt(void)
{
	this->Clean();
	//------------
	CFindCtf2D* pFindCtf2D = new CFindCtf2D;
	m_pvFindCtf2D = pFindCtf2D;
	CInput* pInput = CInput::GetInstance();
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	CLoadImages* pLoadImages = CLoadImages::GetInstance();
	CAsyncSaveImages* pAsyncSaveImages = CAsyncSaveImages::GetInstance();
	CSaveCtfResults* pSaveCtfResults = CSaveCtfResults::GetInstance();
	//-------------------------------------------------------------------
	CCTFTheory aInputCtf;
	float fExtPhase = pInput->m_afExtPhase[0] * 0.017453f;
	aInputCtf.Setup(pInput->m_fKv, pInput->m_fCs,
	   pInput->m_fAmpContrast, pInput->m_fPixelSize,
	   100.0f, fExtPhase);
	pFindCtf2D->Setup1(&aInputCtf);
	//-----------------------------
	bool bPop = true, bRaw = true, bToHost = true;
	m_iNumDone = 0;
	m_pRefPackage = 0L;
	m_iNumPackages = pInputFolder->GetNumPackages();
	//--------------------------------------------
	while(m_iNumDone < m_iNumPackages)
	{	m_pPackage = pLoadImages->GetPackage(bPop);
		if(m_pPackage == 0L)
		{	pLoadImages->WaitForExit(0.01f);
			printf("Wait for micrograph to be loaded.\n\n");
			continue;
		}
		//-----------------------------------------------
		// In case different images have different sizes,
		// If the same, no extra oeverheader.
		//-----------------------------------------------
		pFindCtf2D->Setup2(m_pPackage->m_aiImgSize);
		mProcessPackage(m_iNumDone);
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
	//----------------------------------
	delete pFindCtf2D;
	m_pvFindCtf2D = 0L;
}

void CFindSeriesCtfs::mProcessPackage(int iPackage)
{
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	CFindCtf2D* pFindCtf2D = (CFindCtf2D*)m_pvFindCtf2D;
	//--------------------------------------------------
	bool bRaw = true, bToHost = true, bClean = true, bHalf = true;
	pFindCtf2D->GenHalfSpectrum(m_pPackage->m_pfImage);
	//-------------------------------------------------
	//if(iPackage == 0 || !pInputFolder->IsTomo()) mProcessFull();
	if(iPackage == 0) 
	{	mProcessFull();
	}
	else 
	{	mProcessRefine();
		if(m_pPackage->m_fScore <= 0) mProcessFull();
	}
	//--------------------
	mDisplay();
	//---------
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
	float fDfRange = 5000.0f * pInput->m_fPixelSize 
	   * pInput->m_fPixelSize;
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
	   "azimuth  phase   score\n");
	strcat(acInfo, acBuf1);
	//---------------------
	char acBuf2[128] = {'\0'};
	sprintf(acBuf2, "   %4d  %8.2f  %8.2f  %6.2f %6.2f %9.5f\n",
	   m_pPackage->m_iImgIdx + 1, m_pPackage->m_fDfMin,
	   m_pPackage->m_fDfMax,      m_pPackage->m_fAzimuth,
	   m_pPackage->m_fExtPhase,   m_pPackage->m_fScore);
	strcat(acInfo, acBuf2);
	//---------------------
	printf("%s\n", acInfo);
}
