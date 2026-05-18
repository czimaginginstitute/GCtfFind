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
	CInput* pInput = CInput::GetInstance();
	float fAstRange = pInput->m_fAstRange;
	//---------------------------	
	CFindCtf1D::Do1D();
	mEstAstigmatism();
	//---------------------------
	m_pFindDefocus2D->SetResRange(m_afResRange);
	float fDfMean = (m_fDfMin + m_fDfMax) * 0.5f;
        m_pFindDefocus2D->SetInitVals(fDfMean, m_fAstRatio,
           m_fAstAng, m_fExtPhase);
	//---------------------------
	float afDfRange[] = {m_fDfMax - 3000.0f, m_fDfMax + 3000.0f};
	//afDfRange[0] = fmax(afDfRange[0], 2000.0f);
	afDfRange[0] = fmax(afDfRange[0], -3000.0f);
	//---------------------------
	float afPhaseRange[] = {m_afPhaseRange[0], m_afPhaseRange[1]};
	m_pFindDefocus2D->DoIt(m_gfCtfSpect, afDfRange, afPhaseRange);
	mGetResults();
	//---------------------------
	
	float fPhaseRange = m_afPhaseRange[1] - m_afPhaseRange[0];
	mDoIt(8000.0f, fAstRange, 180.0, fPhaseRange, 5);
	//---------------------------
	fAstRange = fmaxf(fAstRange * 0.5f, 0.05f);
	mDoIt(4000.0f, fAstRange * 0.5f, 60.0f, fPhaseRange * 0.5f, 5);
	//---------------------------
	for(int i=0; i<5; i++)
	{	mDoIt(2000.0f, 0.1f, 10.0f, fPhaseRange * 0.125f, 5);
	}
	m_pFindDefocus2D->CalcCtfRes(m_gfCtfSpect);
	mGetResults();
	
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
{	CInput* pInput = CInput::GetInstance();
	float fMaxAstRatio = pInput->m_fAstRange;
	//---------------------------
	float fDfMean = (m_fDfMin + m_fDfMax) * 0.5f;
	float fMinDf = fDfMean - 0.5f * fDfRange;
	float fMaxDf = fDfMean + 0.5f * fDfRange;
	fMinDf = fmaxf(fMinDf, 2000.0f);
	float fStepDf = (fMaxDf - fMinDf) / 100.0f;
	//---------------------------
	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
           fMinDf, fMaxDf, fStepDf, 0);
	//---------------------------
        m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
           m_fAstAng - 0.5f * fAstAngRange,
           m_fAstAng + 0.5f * fAstAngRange,
           2.0f, 2);
	//---------------------------
	float fMinRatio = m_fAstRatio - 0.5f * fAstMagRange;
	float fMaxRatio = m_fAstRatio + 0.5f * fAstMagRange;
	if(fMinRatio < 0.0f) fMinRatio = 0.0f;
	if(fMaxRatio > fMaxAstRatio) fMaxRatio = fMaxAstRatio;
	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   fMinRatio, fMaxRatio, 0.001, 1);
	//---------------------------
	if(fPhaseRange <= 0.5f)
	{	mGetResults();
		return;
	}
	//---------------------------
        float fMinPhase = m_fExtPhase - fPhaseRange;
        float fMaxPhase = m_fExtPhase + fPhaseRange;
	fMinPhase = fmax(fMinPhase, m_afPhaseRange[0]);
	fMaxPhase = fmin(fMaxPhase, m_afPhaseRange[1]);
	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   fMinPhase, fMaxPhase, 1.0f, 3);
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
	GAstAngle astAngle;
	astAngle.DoIt(m_gfCtfSpect, m_aiCmpSize);
	m_fAstAng = astAngle.m_fAstAng;
	//---------------------------
	GAstRatio astRatio;
	astRatio.DoIt(m_gfCtfSpect, m_aiCmpSize);
	m_fAstRatio = astRatio.m_fAstRatio;
	m_fAstRatio = fminf(m_fAstRatio, 0.1f);
	//---------------------------
	printf("Astigmatism ratio & angle: %.3f  %.2f\n\n",
	   m_fAstRatio, m_fAstAng);
}
