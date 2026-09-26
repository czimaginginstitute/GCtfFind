#include "CUtilInc.h"
#include <string.h>
#include <stdio.h>
#include <Util/Util_LinEqs.h>

using namespace GCTFFind;

CRegSpline3::CRegSpline3(void)
{
     m_iDim = 6;
     m_pfSoln = new float[m_iDim];
     m_pfCoeff = new float[m_iDim * m_iDim];
     m_pfTerms = new float[m_iDim];
}

CRegSpline3::~CRegSpline3(void)
{
     if(m_pfSoln != 0L) delete[] m_pfSoln;
     if(m_pfCoeff != 0L) delete[] m_pfCoeff;
     if(m_pfTerms != 0L) delete[] m_pfTerms;
}

float CRegSpline3::Smooth(float fX)
{
     mCalcTerms(fX);
     float fVal = 0.0f;
     for(int i=0; i<m_iDim; i++)
     {    fVal += (m_pfSoln[i] * m_pfTerms[i]);
     }
     return fVal;
}

void CRegSpline3::DoIt(float* pfDataX, float* pfDataY, int iSize)
{
     int iInterval = iSize / 4;
	 m_fR1 = iInterval;
	 m_fR2 = m_fR1 + iInterval;
	 m_fR2 = m_fR2 + iInterval;
	 mDoIt(pfDataX, pfDataY, iSize);
}

double CRegSpline3::mDoIt(float* pfDataX, float* pfDataY, int iSize)
{
     memset(m_pfSoln, 0, sizeof(float) * m_iDim);
     memset(m_pfCoeff, 0, sizeof(float) * m_iDim * m_iDim);
     //----------------------------------------------------
     float fFact = 1.0f / iSize;
     //-------------------------
     for(int i=0; i<iSize; i++)
     {    mCalcTerms(pfDataX[i]);
          //---------------------
          for(int r=0; r<m_iDim; r++)
          {    for(int c=r; c<m_iDim; c++)
               {    int j = r * m_iDim + c;
                    m_pfCoeff[j] += (m_pfTerms[r] * m_pfTerms[c] * fFact);
               }
               m_pfSoln[r] += (m_pfTerms[r] * pfDataY[i] * fFact);
          }
          for(int r=0; r<m_iDim; r++)
          {    for(int c=0; c<r; c++)
               {    int j = r * m_iDim + c;
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
     {    float fErr = this->Smooth(pfDataX[i]) - pfDataY[i];
          dErr += (fErr * fErr);
     }
     dErr = sqrt(dErr / iSize);
     return dErr;
}

void CRegSpline3::mCalcTerms(float fX)
{
     float fX2 = fX * fX;
     m_pfTerms[0] = 1.0f;
     m_pfTerms[1] = fX;
     //----------------
	 if(fX < m_fR1)
     {	m_pfTerms[2] = fX2;
		m_pfTerms[3] = 0.0f;
        m_pfTerms[4] = 0.0f;
		m_pfTerms[5] = 0.0f;
		return;
     }
     //--------
	 float fXR1 = fX - m_fR1;
	 if(fX < m_fR2)
	 {	m_pfTerms[2] = fX2 - fXR1 * fXR1;
		m_pfTerms[3] = fXR1 * fXR1;
		m_pfTerms[3] = 0.0f;
		m_pfTerms[4] = 0.0f;
		return;
     }
     //--------
	 float fXR2 = fX - m_fR2;
	 if(fX < m_fR3)
	 {	m_pfTerms[2] = fX2 - fXR1 * fXR1;
		m_pfTerms[3] = fXR1 * fXR1 - fXR2 * fXR2;
		m_pfTerms[4] = fXR2 * fXR2;
		m_pfTerms[5] = 0.0f;
		return;
	 }
	 //--------
     float fXR3 = fX - m_fR3;
     m_pfTerms[2] = fX2 - fXR1 * fXR1;
	 m_pfTerms[3] = fXR1 * fXR1 - fXR2 * fXR2;
     m_pfTerms[4] = fXR2 * fXR2 - fXR3 * fXR3;
     m_pfTerms[5] = fXR3 * fXR3;
}
