#include "CAppInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace Thonring::App;

CRotAvg::CRotAvg(void)
{
	m_pfAvg = 0L;
	m_dRad = 4.0 * atan(1.0) / 180.0;
}

CRotAvg::~CRotAvg(void)
{
	if(m_pfAvg != 0L) delete[] m_pfAvg;
}

float* CRotAvg::GetAvg(bool bClean)
{
	float* pfAvg = m_pfAvg;
	if(bClean) m_pfAvg = 0L;
	return pfAvg;
}

void CRotAvg::DoIt(float* pfHalfAmp, int* piSize)
{
	m_pfHalfAmp = pfHalfAmp;
	m_aiAmpSize[0] = piSize[0];
	m_aiAmpSize[1] = piSize[1];
	//-------------------------
	if(m_pfAvg != 0L) delete[] m_pfAvg;
	m_iSize = (piSize[0] < piSize[1]) ? piSize[0] : piSize[1];
	m_pfAvg = new float[m_iSize];
	//---------------------------
	int iCentY = piSize[1] / 2 - 1;
	m_pfAvg[0] = pfHalfAmp[iCentY * piSize[0]];
	//-----------------------------------------
	int iEnd = m_iSize - 1;
	for(int i=1; i<iEnd; i++)
	{	int iSteps = 4 * i;
		if(iSteps > 360) iSteps = 360;
		m_pfAvg[i] = mCalcAvg(i, iSteps);	
	}
}

float CRotAvg::mCalcAvg(int iRadius, int iSteps)
{
	int iCentY = m_aiAmpSize[1] / 2 - 1;
	double dAvg = 0;
	float fStepA = 180.0f / iSteps;
	//-----------------------------
	for(int i=0; i<iSteps; i++)
	{	double dA = (i * fStepA - 90) * m_dRad;
		float fCos = (float)cos(dA);
		float fSin = (float)sin(dA);
		float fX = iRadius * fCos;
		float fY = iRadius * fSin + iCentY;
		dAvg += mInterpolate(fX, fY);
	}
	return (float)(dAvg / iSteps);
}

float CRotAvg::mInterpolate(float fX, float fY)
{
	int iX = (int)fX;
	int iY = (int)fY;
	float fDx = fX - iX;
	float fDy = fY - iY;
	int i11 = iY * m_aiAmpSize[0] + iX;
	int i12 = i11 + m_aiAmpSize[0];
	//-----------------------------
	float fVal = m_pfHalfAmp[i11] * (1 - fDx) * (1 - fDy)
	   	+ m_pfHalfAmp[i12] * (1 - fDx) * fDy
	   	+ m_pfHalfAmp[i11 + 1] * fDx * (1 - fDy)
	   	+ m_pfHalfAmp[i21 + 1] * fDx * fDy;	
	return fVal;
}

