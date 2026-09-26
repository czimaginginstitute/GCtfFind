#include "CUtilInc.h"
#include <string.h>
#include <stdio.h>
#include <Util/Util_LinEqs.h>

using namespace GCTFFind;

CRegSpline2::CRegSpline2(void)
{
    m_iDim = 6;
	m_pfSoln = new float[m_iDim];
	m_pfCoeff = new float[m_iDim * m_iDim];
	m_pfTerms = new float[m_iDim];
}

CRegSpline2::~CRegSpline2(void)
{
	if(m_pfSoln != 0L) delete[] m_pfSoln;
	if(m_pfCoeff != 0L) delete[] m_pfCoeff;
	if(m_pfTerms != 0L) delete[] m_pfTerms;
}

float CRegSpline2::Smooth(float fX)
{
	mCalcTerms(fX);
	float fVal = 0.0f;
	for(int i=0; i<m_iDim; i++)
	{	fVal += (m_pfSoln[i] * m_pfTerms[i]);
	}
	return fVal;
}

void CRegSpline2::SetKnots(float fR1, float fR2)
{
	m_fR1 = fR1;
	m_fR2 = fR2;
}

float CRegSpline2::DoIt(float* pfDataX, float* pfDataY, int iSize)
{
	memset(m_pfSoln, 0, sizeof(float) * m_iDim);
	memset(m_pfCoeff, 0, sizeof(float) * m_iDim * m_iDim);
    //----------------------------------------------------
	float fFact = 1.0f / iSize;
    //-------------------------
    for(int i=0; i<iSize; i++)
    {	mCalcTerms(pfDataX[i]);
    	//---------------------
        for(int r=0; r<m_iDim; r++)
    	{	for(int c=r; c<m_iDim; c++)
    		{	int j = r * m_iDim + c;
            	m_pfCoeff[j] += (m_pfTerms[r] * m_pfTerms[c] * fFact);
			}
			m_pfSoln[r] += (m_pfTerms[r] * pfDataY[i] * fFact);
		}
		for(int r=0; r<m_iDim; r++)
		{	for(int c=0; c<r; c++)
			{	int j = r * m_iDim + c;
            	m_pfCoeff[j] = m_pfCoeff[c*m_iDim+r];
			}
		}
	}
	//--------------------------------------------------
	Util_LinEqs linEqs;
	linEqs.DoIt(m_pfCoeff, m_pfSoln, m_iDim);
	//---------------------------------------
	double dErr = 0.0f;
	for(int i=0; i<iSize; i++)
	{	float fErr = this->Smooth(pfDataX[i]) - pfDataY[i];
		dErr += (fErr * fErr);
	}
	dErr = sqrt(dErr / iSize);
	return (float)dErr;
}

void CRegSpline2::mCalcTerms(float fX)
{
	float fX2 = fX * fX;
	float fX3 = fX * fX2;
	//-------------------
	m_pfTerms[0] = 1.0f;
	m_pfTerms[1] = fX;
	m_pfTerms[2] = fX2;
	//-----------------
	if(fX < m_fR1)
	{	m_pfTerms[3] = fX3;
    	m_pfTerms[4] = 0.0f;
    	m_pfTerms[5] = 0.0f;
    	return;
	}
	//----------
	float fXR1 = fX - m_fR1;
	if(fX < m_fR2)
	{	m_pfTerms[5] = 0.0f;
		m_pfTerms[4] = fXR1 * fXR1 * fXR1;
		m_pfTerms[3] = fX3 - m_pfTerms[4];
		return;
	}
	//----------
	float fXR1_3 = fXR1 * fXR1 * fXR1;
	float fXR2 = fX - m_fR2;
	m_pfTerms[5] = fXR2 * fXR2 * fXR2;
	m_pfTerms[4] = fXR1_3 - m_pfTerms[5];
	m_pfTerms[3] = fX3 - fXR1_3;
}
