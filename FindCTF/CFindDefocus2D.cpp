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
	float fDfMean = m_afNewParam[0];
	float fAstRatio = m_afNewParam[1];
	float fDfMin = fDfMean * (1.0f - fAstRatio);
	return fDfMin;
}

float CFindDefocus2D::GetDfMax(void)
{
	float fDfMean = m_afNewParam[0];
	float fAstRatio = m_afNewParam[1];
	float fDfMax = fDfMean * (1.0f + fAstRatio);
	return fDfMax;
}

float CFindDefocus2D::GetAngle(void)
{
	return m_afNewParam[2];
}

float CFindDefocus2D::GetExtPhase(void)
{
	return m_afNewParam[3];
}

float CFindDefocus2D::GetScore(void)
{
	return m_afNewParam[4];
}

float CFindDefocus2D::GetCtfRes(void)
{
	return m_afNewParam[5];
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
	float fCutOn = (m_aiCmpSize[0] - 1) * 0.01f; 
	float fCutOff = (m_aiCmpSize[0] - 1) * 0.9f;
	//---------------------------
	float fRes1 = m_aiCmpSize[1] * m_pCtfParam->m_fPixelSize;
	float fMinFreq = fRes1 / afResRange[0];
	float fMaxFreq = fRes1 / afResRange[1];
	if(fMinFreq < fCutOn) fMinFreq = fCutOn;
	if(fMaxFreq > fCutOff) fMaxFreq = fCutOff;
	//---------------------------
	m_pGCC2D->SetFreqRange(fMinFreq, fMaxFreq);
}

void CFindDefocus2D::SetBFactor(float fBFactor)
{
	m_fBFactor = fBFactor;
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
{	m_afNewParam[0] = fDfMean;
	m_afNewParam[1] = fAstRatio;
	m_afNewParam[2] = fAstAngle;
	m_afNewParam[3] = fExtPhase;
	m_afNewParam[4] = (float)-1e20;
	m_afNewParam[5] = (float)1e20;
}

void CFindDefocus2D::DoIt
(	float* gfSpect,
	float* pfDfRange,
	float* pfPhaseRange
)
{	m_gfSpect = gfSpect;
	mCalcMetric(pfDfRange, pfPhaseRange);
}

void CFindDefocus2D::RefineParam
(	float* gfSpect,
	float fMinVal, 
	float fMaxVal,
	float fStep,
	int iParam
)
{	m_gfSpect = gfSpect;
	float fRange = fMaxVal - fMinVal;
	if(fRange == 0.0f) return;
	else if(fStep <= 0) return;
	else memcpy(m_afOldParam, m_afNewParam, sizeof(m_afNewParam));
	//---------------------------
	int iNumSteps = (int)(fRange / fStep + 0.5f);
	iNumSteps = iNumSteps / 2 * 2 + 1;
	int iCent = iNumSteps / 2;
	//---------------------------
	float fMaxCC = mCorrelate();
	float fBestVal = m_afNewParam[iParam];
	float fInitVal = fBestVal;
	//---------------------------
	for(int i=0; i<iNumSteps; i++)
	{	m_afNewParam[iParam] = fInitVal + fStep * (i - iCent);
		if(m_afNewParam[iParam] < fMinVal) continue;
		else if(m_afNewParam[iParam] > fMaxVal) continue;
		//-------------------
		float fCC = mCorrelate();
		if(fCC > fMaxCC)
		{	fMaxCC = fCC;
			fBestVal = m_afNewParam[iParam];
		}
	}
	//---------------------------
	if(fMaxCC > m_afNewParam[4]) 
	{	m_afNewParam[iParam] = fBestVal;
		m_afNewParam[4] = fMaxCC;
		if(m_afNewParam[3] > 180) m_afNewParam[3] -= 180.0f;
	}
	else memcpy(m_afNewParam, m_afOldParam, sizeof(m_afOldParam));
}

void CFindDefocus2D::Refine
(	float* gfSpect,
	float fDfRange,
	float fPhaseRange
)
{	float fDfMean = m_afNewParam[0];
	float fMinDf = fDfMean - 0.5f * fDfRange;
	float fMaxDf = fDfMean + 0.5f * fDfRange;
	fMinDf = fmax(fMinDf, 1000.0f);
	this->RefineParam(gfSpect, fMinDf, fMaxDf, 0, 3);
	//---------------------------
	float fExtPhase = m_afNewParam[3];
	float fMinPhase = fExtPhase - 0.5f * fPhaseRange;
	float fMaxPhase = fExtPhase + 0.5f * fPhaseRange;
	fMinPhase = fmaxf(fMinPhase, 0.0f);
	fMaxPhase = fminf(fMaxPhase, 180.0f);
	this->RefineParam(gfSpect, fMinPhase, fMaxPhase, 3, 3);	
}

void CFindDefocus2D::CalcCtfRes(float* gfSpect)
{
	float fDfMean = m_afNewParam[0];
	float fAstRatio = m_afNewParam[1];
	float fExtPhaseRad = m_afNewParam[3] * s_fD2R;
	float fAstRad = m_afNewParam[2] * s_fD2R;
	//---------------------------
	float fDfMin = CFindCtfHelp::CalcDfMin(fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	float fDfMax = CFindCtfHelp::CalcDfMax(fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	//-----------------
	m_aGCalcCtf2D.DoIt(fDfMin, fDfMax, fAstRad, fExtPhaseRad,
	   m_gfCtf2D, m_aiCmpSize);
	//-----------------
	GSpectralCC2D gSpectCC;
	gSpectCC.SetSize(m_aiCmpSize);
	int iShell = gSpectCC.DoIt(m_gfCtf2D, gfSpect);
	//-----------------
	m_afNewParam[5] = m_aiCmpSize[1] * m_pCtfParam->m_fPixelSize / iShell;
}

float CFindDefocus2D::mCalcMetric
(	float* pfDfRange,
        float* pfPhaseRange
)
{	float fDfStep = 100.0f;
        float fPhStep = 1.0f;
        //---------------------------
        m_pGCC2D->SetBFactor(m_fBFactor);
        //---------------------------
        float fBestDF = 0.0f;
        float fBestPH = 0.0f;
        float fBestCC = (float)-1e20;
        for(float p=pfPhaseRange[0]; p<=pfPhaseRange[1]; p+=fPhStep)
        {       m_afNewParam[3] = p;
                for(float f=pfDfRange[0]; f<=pfDfRange[1]; f+=fDfStep)
                {       m_afNewParam[0] = f;
                        float fCC = mCorrelate();
                        if(fCC > fBestCC)
                        {       fBestDF = f;
                                fBestPH = p;
                                fBestCC = fCC;
                        }
                }
        }
        m_afNewParam[0] = fBestDF;
        m_afNewParam[3] = fBestPH;
        m_afNewParam[4] = fBestCC;
	//---------------------------
	this->CalcCtfRes(m_gfSpect);
        float fMetric = 0.01f * m_afNewParam[4] 
		+ 0.99f / (m_afNewParam[5] + 0.0001f);
	return fMetric;
}



float CFindDefocus2D::mCorrelate(void)
{	
	float fDfMean = m_afNewParam[0];
	float fAstRatio = m_afNewParam[1];
	float fAstRad = m_afNewParam[2] * s_fD2R;
	float fExtPhaseRad = m_afNewParam[3] * s_fD2R;
	//---------------------------
	float fDfMin = CFindCtfHelp::CalcDfMin(fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	float fDfMax = CFindCtfHelp::CalcDfMax(fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	//---------------------------
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

