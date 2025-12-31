#include "CFindCTFInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

static float s_fD2R = 0.01745329f;

CFindDefocus2D::CFindDefocus2D(void)
{
	m_gfCtf2D = 0L;
	m_pGCC2D = 0L;
	m_fAstRatio = 0.0f; // (m_fDfMean - fMinDf) / m_fDfMean;
	m_fAstAngle = 0.0f; // degree
}

CFindDefocus2D::~CFindDefocus2D(void)
{
	this->Clean();
}

void CFindDefocus2D::Clean(void)
{
	if(m_gfCtf2D != 0L) 
	{	cudaFree(m_gfCtf2D);
		m_gfCtf2D = 0L;
	}
	if(m_pGCC2D != 0L)
	{	delete m_pGCC2D;
		m_pGCC2D = 0L;
	}
}

float CFindDefocus2D::GetDfMin(void)
{
	float fDfMin = m_fDfMean * (1.0f - m_fAstRatio);
	return fDfMin;
}

float CFindDefocus2D::GetDfMax(void)
{
	float fDfMax = m_fDfMean * (1.0f + m_fAstRatio);
	return fDfMax;
}

float CFindDefocus2D::GetAngle(void)
{
	return m_fAstAngle;
}

float CFindDefocus2D::GetExtPhase(void)
{
	return m_fExtPhase;
}

float CFindDefocus2D::GetScore(void)
{
	return m_fMaxCC;
}

float CFindDefocus2D::GetCtfRes(void)
{
	return m_fCtfRes;
}

void CFindDefocus2D::Setup(CCTFParam* pCtfParam, int* piCmpSize)
{
	this->Clean();
	//---------------------------
	m_pCtfParam = pCtfParam;
	m_aGCalcCtf2D.SetParam(m_pCtfParam);
	//---------------------------
	memcpy(m_aiCmpSize, piCmpSize, sizeof(int) * 2);
	cudaMalloc(&m_gfCtf2D, sizeof(float) 
	   * m_aiCmpSize[0] * m_aiCmpSize[1]);
	//---------------------------
	m_pGCC2D = new GCC2D;
	m_pGCC2D->SetSize(m_aiCmpSize);	
}

void CFindDefocus2D::SetResRange(float afResRange[2])
{
	float fRes1 = m_aiCmpSize[1] * m_pCtfParam->m_fPixelSize;
	float fMinFreq = fRes1 / afResRange[0];
	float fMaxFreq = fRes1 / afResRange[1];
	m_pGCC2D->Setup(fMinFreq, fMaxFreq, 16.0f);
}

//--------------------------------------------------------------------
// 1. DoIt() should be called after CFindDefocus1D::DoIt(), which
//    generates an estimate of m_fDfMean.
//--------------------------------------------------------------------
void CFindDefocus2D::SetInitVals
(	float fDfMean,
	float fAstRatio,
	float fAstAngle,
	float fExtPhase
)
{	m_fDfMean = fDfMean;
	m_fAstRatio = fAstRatio;
	m_fAstAngle = fAstAngle;
	m_fExtPhase = fExtPhase;
	m_fMaxCC = (float)-1e20;
}

void CFindDefocus2D::DoIt
(	float* gfSpect,
	float* pfDfRange,
	float* pfPhaseRange
)
{	m_gfSpect = gfSpect;
	float fDfStep = 200.0f;
	float fPhaseStep = 2.0f;
	//---------------------------
	int iDfSteps = (int)((pfDfRange[1] - pfDfRange[0]) / fDfStep);
	int iPhaseSteps = (int)((pfPhaseRange[1] - pfPhaseRange[0])
	   / fPhaseStep);
	if(iDfSteps < 1) iDfSteps = 1;
	if(iPhaseSteps < 1) iPhaseSteps = 1;
	//---------------------------
	float fMaxCC = (float)-1e20;
	float fBestDf = 0.0f;
	float fBestPhase = 0.0f;
	for(int j=0; j<iPhaseSteps; j++)
	{	float fPhase = m_fExtPhase + (j - iPhaseSteps / 2) 
		   * fPhaseStep;
		fPhase = fmax(fPhase, pfPhaseRange[0]);
		fPhase = fmin(fPhase, pfPhaseRange[1]);
		//-------------------
		for(int i=0; i<iDfSteps; i++)
		{	float fDf = m_fDfMean + (i - iDfSteps / 2)
			   * fDfStep;
			fDf = fmax(fDf, pfDfRange[0]);
			fDf = fmin(fDf, pfDfRange[1]);
			//------------------
			//if(fDf < 2000.0f) fDf = 2000.0f;
			float fCC = mCorrelate(fDf, m_fAstRatio,
			   m_fAstAngle, fPhase);
			if(fCC <= fMaxCC) continue;
			fMaxCC = fCC;
			fBestDf = fDf;
			fBestPhase = fPhase;
		}
	}
	if(fMaxCC <= m_fMaxCC) return;
	m_fDfMean = fBestDf;
	m_fExtPhase = fBestPhase;
	m_fMaxCC = fMaxCC;	
}

void CFindDefocus2D::RefineParam
(	float* gfSpect,
	float fMinVal, 
	float fMaxVal,
	int iParam,
	int iIterations
)
{	m_gfSpect = gfSpect;
	float fRange = fMaxVal - fMinVal;
	if(fRange == 0.0f) return;
	//---------------------------
	int iNumSteps = 31;
	int iCent = iNumSteps / 2;
	float fStep = fRange / (iNumSteps - 1);
	//---------------------------
	float* pfVals = new float[iNumSteps];
	float* pfCCs = new float[iNumSteps];
	float fMaxCC = (float)-1e20;
	float fBestVal = 0.0f;
	//---------------------------
	float afVal[] = {m_fDfMean, m_fAstRatio, m_fAstAngle, m_fExtPhase};
	float fInitVal = afVal[iParam];
	//---------------------------
	for(int j=0; j<iIterations; j++)
	{	for(int i=0; i<iNumSteps; i++)
		{	afVal[iParam] = fInitVal + fStep * (i - iCent);
			afVal[iParam] = fmaxf(afVal[iParam], fMinVal);
			afVal[iParam] = fminf(afVal[iParam], fMaxVal);
			pfVals[i] = afVal[iParam];
			//-----------
			float fCC = mCorrelate(afVal[0], afVal[1],
		   	   afVal[2], afVal[3]);
			pfCCs[i] = fCC;
			//-----------
			if(fCC <= fMaxCC) continue;
			fMaxCC = fCC;
			fBestVal = afVal[iParam];
		}
		//-------------------
		afVal[iParam] = mFitNewVal(pfVals, pfCCs, iNumSteps);	
		afVal[iParam] = fmaxf(afVal[iParam], fMinVal);
		afVal[iParam] = fminf(afVal[iParam], fMaxVal);
		float fCC = mCorrelate(afVal[0], afVal[1], afVal[2], afVal[3]); 
		//-------------------
		if(fCC > fMaxCC)
		{	fMaxCC = fCC;
			fBestVal = afVal[iParam];
		}
		if(fMaxCC <= m_fMaxCC) continue;
		m_fMaxCC = fMaxCC;
		fInitVal = fBestVal;
	}
	if(iParam == 0) m_fDfMean = fInitVal;
	else if(iParam == 1) m_fAstRatio = fInitVal;
	else if(iParam == 2) m_fAstAngle = fInitVal;
	else if(iParam == 3) m_fExtPhase = fInitVal;
	//---------------------------
	if(pfVals != 0L) delete[] pfVals;
	if(pfCCs != 0L) delete[] pfCCs;
}

void CFindDefocus2D::Refine
(	float* gfSpect,
	float fDfRange,
	float fPhaseRange
)
{	float fMinDf = m_fDfMean - 0.5f * fDfRange;
	float fMaxDf = m_fDfMean + 0.5f * fDfRange;
	fMinDf = fmax(fMinDf, 1000.0f);
	this->RefineParam(gfSpect, fMinDf, fMaxDf, 0, 3);
	//---------------------------
	float fMinPhase = m_fExtPhase - 0.5f * fPhaseRange;
	float fMaxPhase = m_fExtPhase + 0.5f * fPhaseRange;
	fMinPhase = fmaxf(fMinPhase, 0.0f);
	fMaxPhase = fminf(fMaxPhase, 150.0f);
	this->RefineParam(gfSpect, fMinPhase, fMaxPhase, 3, 3);	
}

void CFindDefocus2D::CalcCtfRes(float* gfSpect)
{
	float fExtPhaseRad = m_fExtPhase * s_fD2R;
	float fAstRad = m_fAstAngle * s_fD2R;
	//-----------------
	float fDfMin = CFindCtfHelp::CalcDfMin(m_fDfMean, m_fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	float fDfMax = CFindCtfHelp::CalcDfMax(m_fDfMean, m_fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	//-----------------
	m_aGCalcCtf2D.DoIt(fDfMin, fDfMax, fAstRad, fExtPhaseRad,
	   m_gfCtf2D, m_aiCmpSize);
	//-----------------
	GSpectralCC2D gSpectCC;
	gSpectCC.SetSize(m_aiCmpSize);
	int iShell = gSpectCC.DoIt(m_gfCtf2D, gfSpect);
	//-----------------
	m_fCtfRes = m_aiCmpSize[1] * m_pCtfParam->m_fPixelSize / iShell;
}


float CFindDefocus2D::mCorrelate
(	float fDfMean,
 	float fAstRatio,  // (fmax - fmin) / (fmax + fmin)
	float fAstAngle,  // degree 
	float fExtPhase   // degree
)
{	float fExtPhaseRad = fExtPhase * s_fD2R;
	float fAstRad = fAstAngle * s_fD2R;
	//---------------------------------
	float fDfMin = CFindCtfHelp::CalcDfMin(fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	float fDfMax = CFindCtfHelp::CalcDfMax(fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	//-----------------------------
	m_aGCalcCtf2D.DoIt(fDfMin, fDfMax, fAstRad, fExtPhaseRad, 
	   m_gfCtf2D, m_aiCmpSize);
	float fCC = m_pGCC2D->DoIt(m_gfCtf2D, m_gfSpect);
	return fCC;
}

void CFindDefocus2D::mGetRange
(	float fCentVal,
	float fRange,
	float* pfMinMax,
	float* pfRange
)
{	pfRange[0] = fCentVal - fRange * 0.5f;
	pfRange[1] = fCentVal + fRange * 0.5f;
	if(pfRange[0] < pfMinMax[0]) pfRange[0] = pfMinMax[0];
	if(pfRange[1] > pfMinMax[1]) pfRange[1] = pfMinMax[1];
}

float CFindDefocus2D::mFitNewVal(float* x, float* y, int n)
{
	float fMax = (float)-1e20;
	int iMax = -1;
	for(int i = 0; i < n; i++)
	{	if(y[i] <= fMax) continue;
		fMax = y[i];
		iMax = i;
	}
	int iStart = iMax - 1;
	if(iStart < 0) iStart = 0;
	int iEnd = iStart + 3;
	if(iEnd > n) iEnd = n;
	iStart = iEnd - 3;
	//---------------------------
	double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
	double sum_y = 0, sum_xy = 0, sum_x2y = 0;
	for (int i = iStart; i < iEnd; i++) 
	{	sum_x   += x[i];
		sum_x2  += x[i] * x[i];
		sum_x3  += x[i] * x[i] * x[i];
		sum_x4  += x[i] * x[i] * x[i] * x[i];
		sum_y   += y[i];
		sum_xy  += x[i] * y[i];
		sum_x2y += x[i] * x[i] * y[i];
    	}
	//---------------------------
	double D, Da, Db, Dc;
	double a, b, c;
	D  = sum_x4 * (sum_x2 * n - sum_x * sum_x)
		- sum_x3 * (sum_x3 * n - sum_x * sum_x2)
		+ sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2);
	Da = sum_x2y * (sum_x2 * n - sum_x * sum_x)
		- sum_x3  * (sum_xy * n - sum_x * sum_y)
		+ sum_x2  * (sum_xy * sum_x - sum_x2 * sum_y);
	Db = sum_x4 * (sum_xy * n - sum_x * sum_y)
		- sum_x2y * (sum_x3 * n - sum_x * sum_x2)
		+ sum_x2  * (sum_x3 * sum_y - sum_xy * sum_x2);
	Dc = sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy)
		- sum_x3 * (sum_x3 * sum_y - sum_x2 * sum_xy)
		+ sum_x2y * (sum_x3 * sum_x - sum_x2 * sum_x2);
	D += 1e-30;
	a = Da / D;
	b = Db / D;
	c = Dc / D;
	//---------------------------
	float fNewVal = (float)(b / (2.0 * a + 1e-30));
	return fNewVal;	
}

