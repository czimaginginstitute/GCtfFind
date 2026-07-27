#include "CMainInc.h"
#include "Util/CUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace GCTFFind;

CSearchRanges* CSearchRanges::m_pInstance = 0L;

CSearchRanges* CSearchRanges::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CSearchRanges;
	return m_pInstance;
}

void CSearchRanges::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CSearchRanges::CSearchRanges(void)
{
}

CSearchRanges::~CSearchRanges(void)
{
}

void CSearchRanges::GetDefocus(float* pfRange)
{
	memcpy(pfRange, m_afDefocus, sizeof(float) * 2);
}

void CSearchRanges::GetAstRatio(float* pfRange)
{
	memcpy(pfRange, m_afAstRatio, sizeof(float) * 2);
}

void CSearchRanges::GetAstAngle(float* pfRange)
{
	memcpy(pfRange, m_afAstAngle, sizeof(float) * 2);
}

void CSearchRanges::GetExtPhase(float* pfRange)
{
	memcpy(pfRange, m_afExtPhase, sizeof(float) * 2);
}

float CSearchRanges::GetDefocus(bool bCentVal)
{
	if(bCentVal) return (m_afDefocus[0] + m_afDefocus[1]) * 0.5f;
	else return (m_afDefocus[1] - m_afDefocus[0]);
}

float CSearchRanges::GetAstRatio(bool bCentVal)
{
	if(bCentVal) return (m_afAstRatio[0] + m_afAstRatio[1]) * 0.5f;
	else return (m_afAstRatio[1] - m_afAstRatio[0]);
}

float CSearchRanges::GetAstAngle(bool bCentVal)
{
	if(bCentVal) return (m_afAstAngle[0] + m_afAstAngle[1]) * 0.5f;
	else return (m_afAstAngle[1] - m_afAstAngle[0]);
}

float CSearchRanges::GetExtPhase(bool bCentVal)
{
	if(bCentVal) return (m_afExtPhase[0] + m_afExtPhase[1]) * 0.5f;
	else return (m_afExtPhase[1] - m_afExtPhase[0]);
}

bool CSearchRanges::bDefocus(void)
{
	float fDelta = m_afDefocus[1] - m_afDefocus[0];
	if(fabsf(fDelta) < 10.0f) return false;
	else return true;
}

bool CSearchRanges::bAstRatio(void)
{
	float fDelta = m_afAstRatio[1] - m_afAstRatio[0];
	if(fDelta < 0.0001f) return false;
	else return true;
}

bool CSearchRanges::bAstAngle(void)
{
	float fDelta = m_afAstAngle[1] - m_afAstAngle[0];
	if(fDelta < 1.0f) return false;
	else return true;
}

bool CSearchRanges::bExtPhase(void)
{
	float fDelta = m_afExtPhase[1] - m_afExtPhase[0];
	if(fDelta < 1.0f) return false;
	else return true;
}

void CSearchRanges::Setup(void)
{
	mSetDfRange();
	mSetAstRange();
	mSetPhaseRange();
}

void CSearchRanges::CheckDefocus(float* pfDfRange)
{
	pfDfRange[0] = fmax(pfDfRange[0], m_afDefocus[0]);
        pfDfRange[1] = fmin(pfDfRange[1], m_afDefocus[1]);
}

void CSearchRanges::CheckAstRatio(float* pfAstRatioRange)
{
	pfAstRatioRange[0] = fmax(pfAstRatioRange[0], m_afAstRatio[0]);
	pfAstRatioRange[1] = fmin(pfAstRatioRange[1], m_afAstRatio[1]);
}

void CSearchRanges::CheckAstAngle(float* pfAstAngleRange)
{
	pfAstAngleRange[0] = fmax(pfAstAngleRange[0], m_afAstAngle[0]);
	pfAstAngleRange[1] = fmin(pfAstAngleRange[1], m_afAstAngle[1]);
}

void CSearchRanges::CheckExtPhase(float* pfExtPhaseRange)
{
	pfExtPhaseRange[0] = fmax(pfExtPhaseRange[0], m_afExtPhase[0]);
	pfExtPhaseRange[1] = fmin(pfExtPhaseRange[1], m_afExtPhase[1]);
}

void CSearchRanges::mSetDfRange(void)
{
	CInput* pInput = CInput::GetInstance();
	float fPixSize2 = pInput->m_fPixSize * pInput->m_fPixSize;
	if(pInput->m_afDefocus[0] == 0 && pInput->m_afDefocus[1] == 0)
	{	float fPixSize2 = pInput->m_fPixSize 
		   * pInput->m_fPixSize;
		float fRange = 25000.0f * fPixSize2;
		m_afDefocus[0] = -5000.0f * fPixSize2;
		m_afDefocus[1] = m_afDefocus[0] + fRange;
	}
	else
	{	float fDelta = pInput->m_afDefocus[1] * 0.5f;
		m_afDefocus[0] = pInput->m_afDefocus[0] - fDelta;
		m_afDefocus[1] = pInput->m_afDefocus[0] + fDelta;
	}
}

void CSearchRanges::mSetAstRange(void)
{
	CInput* pInput = CInput::GetInstance();
	float fDelta = pInput->m_afAstRatio[1] * 0.5f;
	m_afAstRatio[0] = pInput->m_afAstRatio[0] - fDelta;
	m_afAstRatio[1] = pInput->m_afAstRatio[0] + fDelta;
	m_afAstRatio[0] = fmax(m_afAstRatio[0], 0.0f);
	m_afAstRatio[1] = fmax(m_afAstRatio[1], 0.0f);
	//---------------------------
	fDelta = pInput->m_afAstAngle[1] * 0.5f;
	m_afAstAngle[0] = pInput->m_afAstAngle[0] - fDelta;
	m_afAstAngle[1] = pInput->m_afAstAngle[0] + fDelta;
}

void CSearchRanges::mSetPhaseRange(void)
{
	CInput* pInput = CInput::GetInstance();
	float fDelta = 0.5f * pInput->m_afExtPhase[1];
	m_afExtPhase[0] = pInput->m_afExtPhase[0] - fDelta;
	m_afExtPhase[1] = pInput->m_afExtPhase[0] + fDelta;
	m_afExtPhase[0] = fmax(m_afExtPhase[0], 0.0f);
	m_afExtPhase[1] = fmin(m_afExtPhase[1], 150.0f);
}



