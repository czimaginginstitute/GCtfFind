#include "CFindCTFInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace GCTFFind;

CFitParam* CFitParam::m_pInstance = 0L;

CFitParam* CFitParam::GetInstance(void)
{
	if(m_pInstance == 0L) m_pInstance = new CFitParam;
	return m_pInstance;
}

void CFitParam::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CFitParam::CFitParam(void)
{
	m_fPixSize = 1.2f;
	m_afResRange[0] = 25.0f;
	m_afResRange[1] = 4.0f;
	m_fBFactor = 1.0f;
	m_bIceRing1 = false;
	m_bIceRing2 = false;
}

CFitParam::~CFitParam(void)
{
}

void CFitParam::GetIceRange(float* pfIceRange)
{
	if(m_bIceRing1 && m_bIceRing2)
	{	pfIceRange[0] = m_fPixSize / 4.0f;
		pfIceRange[1] = m_fPixSize / 3.4f;
	}
	else if(m_bIceRing1)
	{	pfIceRange[0] = m_fPixSize / 4.0f;
		pfIceRange[1] = m_fPixSize / 3.8f;
	}
	else if(m_bIceRing2)
	{	pfIceRange[0] = m_fPixSize / 3.6f;
		pfIceRange[1] = m_fPixSize / 3.4f;
	}
	else
	{	pfIceRange[0] = 1.0f;
		pfIceRange[1] = 2.0f;
	}
}
