#include "CMainInc.h"
#include "FindCTF/CFindCTFInc.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace GCTFFind;

CCtfPackage::CCtfPackage(void)
{
	memset(m_acMrcFileName, 0, sizeof(m_acMrcFileName));
	m_iImgIdx = -1;
	m_fTilt = 0.0;
	m_fDfMin = 0.0f;
	m_fDfMax = 0.0f;
	m_fAzimuth = 0.0f;
	m_fExtPhase = 0.0f;
	m_fScore = 0.0f;
	m_fCtfRes = 0.0f;
	m_pfImage = 0L;
	m_pfHalfSpect = 0L;
	m_pfFullSpect = 0L;
}

CCtfPackage::~CCtfPackage(void)
{
	this->Clean();
}

void CCtfPackage::Clean(void)
{
	this->CleanSpects();
	//------------------
	if(m_pfImage != 0L) 
	{	delete[] m_pfImage;
		m_pfImage = 0L;
	}
}

void CCtfPackage::CleanSpects(void)
{
	if(m_pfHalfSpect != 0L)
	{	delete[] m_pfHalfSpect;
		m_pfHalfSpect = 0L;
	}
	if(m_pfFullSpect != 0L)
	{	delete[] m_pfFullSpect;
		m_pfFullSpect = 0L;
	}
}

