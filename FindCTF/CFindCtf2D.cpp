#include "CFindCTFInc.h"
#include "../CMainInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

CFindCtf2D::CFindCtf2D(void)
{
	m_pFindDefocus2D = 0L;
}

CFindCtf2D::~CFindCtf2D(void)
{
	this->Clean();
}

void CFindCtf2D::Clean(void)
{
	if(m_pFindDefocus2D != 0L) 
	{	delete m_pFindDefocus2D;
		m_pFindDefocus2D = 0L;
	}
	CFindCtf1D::Clean();
}

void CFindCtf2D::Setup1(CCTFTheory* pCtfTheory)
{
	this->Clean();
	CFindCtf1D::Setup1(pCtfTheory);
	//--------------------------
	m_pFindDefocus2D = new CFindDefocus2D;
	CCTFParam* pCtfParam = m_pCtfTheory->GetParam(false);
	m_pFindDefocus2D->Setup(pCtfParam, m_aiCmpSize);
	m_pFindDefocus2D->SetResRange(m_afResRange);
}

void CFindCtf2D::Do2D(void)
{
	CFindCtf1D::Do1D();
	mEstAstigmatism();
	//---------------------------
	float fDfMean = (m_fDfMin + m_fDfMax) * 0.5f;
        m_pFindDefocus2D->SetInitVals(fDfMean, m_fAstRatio,
           m_fAstAng, m_fExtPhase);
	//---------------------------
	CSearchRanges* pSeaRanges = CSearchRanges::GetInstance();
	float fDfRange = pSeaRanges->GetDefocus(false) * 0.5f;
	float afDfRange[] = {m_fDfMax - fDfRange, m_fDfMax + fDfRange};
	pSeaRanges->CheckDefocus(afDfRange);
	//---------------------------
	float afPhaseRange[2] = {0.0f};
	pSeaRanges->GetExtPhase(afPhaseRange);
	//---------------------------
	m_pFindDefocus2D->SetResRange(m_afResRange);
	m_pFindDefocus2D->DoIt(m_gfCtfSpect, afDfRange, afPhaseRange);
	mGetResults();
	//---------------------------
	mDoIt(4000.0f, 0.1f, 60.0f, 60.0f, 5);
	m_pFindDefocus2D->CalcCtfRes(m_gfCtfSpect);
	mGetResults();
	printf("Astigmatism: %f  %f\n", m_fAstRatio, m_fAstAng);
}

void CFindCtf2D::Refine
(	float afDfMean[2],
	float afAstRatio[2],
	float afAstAngle[2],
	float afExtPhase[2]
)
{	m_pFindDefocus2D->SetResRange(m_afResRange);
	m_pFindDefocus2D->SetInitVals(afDfMean[0], afAstRatio[0],
	   afAstAngle[0], afExtPhase[0]);
	//---------------------------
	m_pFindDefocus2D->Refine(m_gfCtfSpect, afDfMean[1], afExtPhase[1]);
	m_pFindDefocus2D->CalcCtfRes(m_gfCtfSpect);
	mGetResults();
}

void CFindCtf2D::mDoIt
(	float fDfRange,
 	float fAstMagRange,
	float fAstAngRange, 
	float fPhaseRange, 
	int iIterations
)
{	CSearchRanges* pSeaRanges = CSearchRanges::GetInstance();
	//---------------------------
	float afDfRange[2] = {0.0f};	
	float fDfMean = (m_fDfMin + m_fDfMax) * 0.5f;
	afDfRange[0] = fDfMean - 0.5f * fDfRange;
	afDfRange[1] = fDfMean + 0.5f * fDfRange;
	pSeaRanges->CheckDefocus(afDfRange);
	float fRange = afDfRange[1] - afDfRange[0];
	if(fRange >= 1000.0)
	{	float fStepDf = fRange / 100.0f;
		m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
           	   afDfRange[0], afDfRange[1], fStepDf, 0);
	}
	//---------------------------
	float afAngleRange[2] = {0.0f};
	afAngleRange[0] = m_fAstAng - 0.5f * fAstAngRange;
	afAngleRange[1] = m_fAstAng + 0.5f * fAstAngRange;
	pSeaRanges->CheckAstAngle(afAngleRange);
	fRange = afAngleRange[1] - afAngleRange[0];
	if(fRange > 6.0f)
	{	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
          	   afAngleRange[0], afAngleRange[1], 
           	   1.0f, 2);
	}
	//---------------------------
	float afRatioRange[2] = {0.0f};
	afRatioRange[0] = m_fAstRatio - 0.5f * fAstMagRange;
	afRatioRange[1] = m_fAstRatio + 0.5f * fAstMagRange;
	pSeaRanges->CheckAstRatio(afRatioRange);
	fRange = afRatioRange[1] - afRatioRange[0];
	if(fRange > 0.01f)
	{	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   	   afRatioRange[0], afRatioRange[1], 
		   0.001, 1);
	}
	//---------------------------
	float afPhaseRange[2] = {0.0f};
        afPhaseRange[0] = m_fExtPhase - fPhaseRange;
        afPhaseRange[1] = m_fExtPhase + fPhaseRange;
	pSeaRanges->CheckExtPhase(afPhaseRange);
	fRange = afPhaseRange[1] - afPhaseRange[0];
	if(fRange > 5.0f)
	{	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   	afPhaseRange[0], afPhaseRange[1], 
		1.0f, 3);
	}
	//---------------------------
        mGetResults();
}

void CFindCtf2D::mGetResults(void)
{
	m_fDfMin = m_pFindDefocus2D->GetDfMin();
	m_fDfMax = m_pFindDefocus2D->GetDfMax();
	m_fAstAng = m_pFindDefocus2D->GetAngle();
	m_fExtPhase = m_pFindDefocus2D->GetExtPhase();
	m_fScore = m_pFindDefocus2D->GetScore();
	m_fCtfRes = m_pFindDefocus2D->GetCtfRes();
}

void CFindCtf2D::mEstAstigmatism(void)
{
	CSearchRanges* pSeaRanges = CSearchRanges::GetInstance();
	bool bCentVal = true;
	m_fAstRatio = pSeaRanges->GetAstRatio(bCentVal);
	m_fAstAng = pSeaRanges->GetAstAngle(bCentVal);
	//---------------------------
	if(pSeaRanges->bAstAngle())
	{	GAstAngle astAngle;
		astAngle.DoIt(m_gfCtfSpect, m_aiCmpSize);
		m_fAstAng = astAngle.m_fAstAng;
		//-------------------
		float afRange[2] = {0.0f};
		pSeaRanges->GetAstAngle(afRange);
		if(m_fAstAng < afRange[0]) m_fAstAng += 180.0f;
		if(m_fAstAng > afRange[1]) m_fAstAng -= 180.0f;
		//-------------------
		printf("Ast angle init est: %.2f\n", m_fAstAng);
	}
	//---------------------------
	if(pSeaRanges->bAstRatio())
	{	GAstRatio astRatio;
		astRatio.DoIt(m_gfCtfSpect, m_aiCmpSize);
		m_fAstRatio = astRatio.m_fAstRatio;
		m_fAstRatio = fminf(m_fAstRatio, 0.1f);
		printf("Ast ratio init est: %.3f\n", m_fAstRatio);
	}
}
