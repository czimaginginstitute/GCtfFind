#include "CFindCTFInc.h"
#include "../Util/CUtilInc.h"
#include <CuUtilFFT/GFFT1D.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

CCalcBackground::CCalcBackground(void)
{
     m_gfBackground = 0L;
}

CCalcBackground::~CCalcBackground(void)
{
     if(m_gfBackground != 0L) cudaFree(m_gfBackground);
}

float* CCalcBackground::GetBackground(bool bClean)
{
     float* gfBackground = m_gfBackground;
     if(bClean) m_gfBackground = 0L;
     return gfBackground;
}

void CCalcBackground::DoSpline
(	float* pfSpectrum,
	int iSize,
	float fPixelSize
)
{	m_iSize = iSize;
	m_fPixelSize = fPixelSize;
    //------------------------
    size_t tBytes = sizeof(float) * m_iSize;
    float* pfRawSpectrum = new float[m_iSize];
    cudaMemcpy(pfRawSpectrum, pfSpectrum, tBytes, cudaMemcpyDefault);
    //---------------------------------------------------------------
	float* pfBackground = new float[m_iSize];
	memset(pfBackground, 0, sizeof(float) * m_iSize);
	//-----------------------------------------------
	m_iStart = m_iSize / 10; //mFindStart(pfRawSpectrum);
	//-----------------------------------
	float* pfDataX = new float[m_iSize];
	for(int i=0; i<m_iSize; i++) pfDataX[i] = i;
	//------------------------------------------
	CRegSpline2 regSpline;
	int iNewSize = m_iSize - m_iStart;
	float fR1 = pfDataX[iNewSize / 3];
	float fR2 = pfDataX[iNewSize * 2 / 3];
	regSpline.SetKnots(fR1, fR2);
	regSpline.DoIt(pfDataX+m_iStart, pfRawSpectrum+m_iStart, iNewSize);
	//-----------------------------------------------------------------
	for(int i=m_iStart; i<m_iSize; i++)
	{	pfBackground[i] = regSpline.Smooth(pfDataX[i]);
	}
	if(pfDataX != 0L) delete[] pfDataX;
	//---------------------------------
	int iBoxSize = m_iSize / 15;
	for(int i=(m_iStart+iBoxSize/2); i<m_iSize; i++)
	{	int jStart = i - iBoxSize / 2;
		int jEnd = i + iBoxSize / 2;
		float fMean = 0.0f;
		for(int j=jStart; j<jEnd; j++)
		{	int k = (j < m_iSize) ? j : 2 * m_iSize - 1 -j;
			float fVal = pfRawSpectrum[k] - pfBackground[k];
			fMean += fVal;
		}
		fMean /= (jEnd - jStart);
		pfBackground[i] += fMean;
	}
	//---------------------------
	if(m_gfBackground != 0L) cudaFree(m_gfBackground);
	cudaMalloc(&m_gfBackground, tBytes);
	cudaMemcpy(m_gfBackground, pfBackground, tBytes, cudaMemcpyDefault);
	//------------------------------------------------------------------
	delete[] pfBackground;
	delete[] pfRawSpectrum;
}

int CCalcBackground::mFindStart(float* pfSpectrum)
{
	float* pfData = new float[m_iSize];
	for(int i=0; i<m_iSize; i++) pfData[i] = i;
	//-----------------------------------------
	CRegSpline2 regSpline;
	float fR1 = m_iSize / 3.0f;
	float fR2 = 2.0f * fR1;
	regSpline.SetKnots(fR1, fR2);
	float fErr1 = regSpline.DoIt
	(  pfData+1, pfSpectrum+1, m_iSize-1
	);
	//----------------------------------
	int iBestStart = 2;
	int iEnd = m_iSize / 4;
	for(int i=2; i<iEnd; i++)
	{	int iSize = m_iSize - i;
		fR1 = iSize / 3.0f;
		fR2 = 2.0f * fR1;
		regSpline.SetKnots(fR1, fR2);
		float fErr = regSpline.DoIt(pfData+i, pfSpectrum+i, iSize);
		float fDiff = fErr / fErr1;
		if(fDiff > 0.1f) continue;
		iBestStart = i;
		break;
	}
	if(pfData != 0L) delete[] pfData;
	float fReslotion = m_iSize * 2.0f * m_fPixelSize / iBestStart;
	int iBestStart1 = (int)(m_iSize * 2 * m_fPixelSize / 60.0f);
	if(iBestStart1 > (m_iSize / 10)) iBestStart1 = m_iSize / 10;
	//-----------------------------------------------------------
	if(iBestStart < iBestStart1) iBestStart = iBestStart1;
	return iBestStart;
}

float* CCalcBackground::mLinearFit
(	float* pfData,
	int iStart,
	int iEnd
)
{	double dX = 0, dY = 0, dX2 = 0, dXY = 0;
	for(int i=iStart; i<iEnd; i++)
	{	dX += i;
		dX2 += (i * i);
		dY += pfData[i];
		dXY += (i * pfData[i]);
	}
	int iCount = iEnd - iStart;
	dX /= iCount;
	dY /= iCount;
	dX2 /= iCount;
	dXY /= iCount;
	double dSlope = (dX * dY - dXY) / (dX * dX - dX2);
	double dIntercept = dY - dSlope * dX;
	//-------------------------------
	float* pfRes = new float[2];
	pfRes[0] = (float)dSlope;
	pfRes[1] = (float)dIntercept;
	return pfRes;
}
