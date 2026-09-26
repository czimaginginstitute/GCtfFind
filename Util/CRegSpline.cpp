#include "CUtilInc.h"
#include <string.h>
#include <stdio.h>
#include <Util/Util_LinEqs.h>

using namespace GCTFFind;

CRegSpline::CRegSpline(void)
{
     m_iDim = 5;
     m_pfSoln = new float[m_iDim];
     m_pfCoeff = new float[m_iDim * m_iDim];
     m_pfTerms = new float[m_iDim + 1];
}

CRegSpline::~CRegSpline(void)
{
     if(m_pfSoln != 0L) delete[] m_pfSoln;
     if(m_pfCoeff != 0L) delete[] m_pfCoeff;
     if(m_pfTerms != 0L) delete[] m_pfTerms;
}

float CRegSpline::Smooth(float fX)
{
     mCalcTerms(fX, m_fR);
     if(fX < m_fR)
     {    m_pfTerms[4] = 0.0f;
     }
     else
     {    m_pfTerms[3] = m_pfTerms[4];
          m_pfTerms[4] = m_pfTerms[5];
     }
     float fVal = 0.0f;
     for(int i=0; i<m_iDim; i++)
     {    fVal += (m_pfSoln[i] * m_pfTerms[i]);
     }
     return fVal;
}

void CRegSpline::DoIt(float* pfDataX, float* pfDataY, int iSize)
{
     float* pfBestSoln = new float[m_iDim];
     float fBestR = 0.0f;
     double dMinErr = 1e30;
     //--------------------
     int iStart = iSize / 3;
     int iEnd = iSize - iStart;
     for(int i=iStart; i<=iEnd; i++)
     {    float fErr = mDoIt(pfDataX, pfDataY, iSize, i);
          //printf("%d  %e\n", i, fErr);
          if(fErr >= dMinErr) continue;
          //---------------------------
          dMinErr = fErr;
          fBestR = m_fR;
          memcpy(pfBestSoln, m_pfSoln, sizeof(float) * m_iDim);
     }
     //--------------------------------------------------------
     m_fR = fBestR;
     memcpy(m_pfSoln, pfBestSoln, sizeof(float) * m_iDim);
     if(pfBestSoln != 0L) delete[] pfBestSoln;
}

float CRegSpline::mDoIt(float* pfDataX, float* pfDataY, int iSize, int iR)
{
     memset(m_pfSoln, 0, sizeof(float) * m_iDim);
     memset(m_pfCoeff, 0, sizeof(float) * m_iDim * m_iDim);
     //----------------------------------------------------
     m_fR = pfDataX[iR];
     float fFact = 1.0f / iSize;
     //-------------------------
     for(int i=0; i<iSize; i++)
     {    mCalcTerms(pfDataX[i], m_fR);
          if(pfDataX[i] < m_fR)
          {    m_pfTerms[4] = 0.0f;
          }
          else
          {    m_pfTerms[3] = m_pfTerms[4];
               m_pfTerms[4] = m_pfTerms[5];
          }
          //-------------------------------
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
     return (float)dErr;
}

void CRegSpline::mCalcTerms(float fX, float fR)
{
     float fX2 = fX * fX;
     m_pfTerms[0] = 1.0f;
     m_pfTerms[1] = fX;
     m_pfTerms[2] = fX2;
     m_pfTerms[3] = fX2 * fX;
     //----------------------
     float fXR = fX - fR;
     float fXR2 = fXR * fXR;
     m_pfTerms[5] = fXR2 * fXR;
     m_pfTerms[4] = m_pfTerms[3] - m_pfTerms[5];
}

