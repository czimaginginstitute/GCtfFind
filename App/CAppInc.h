#pragma once
#include "../MrcUtil/CMrcUtilInc.h"
#include <Util/Util_Thread.h>
#include <Utilfft/CUtilfftInc.h>
#include <cufft.h>

namespace Thonring
{
namespace App
{
//---------------------------------------------------------
class CAppParam
{
public:
	CAppParam(void)
	{	m_fBFactor = 250.0f;
		m_iIterations = 5;
		m_fTol = 0.5f;
		m_fFourierBin = 1.0f;
		m_aiPatches[0] = 1;
		m_aiPatches[1] = 1;
		m_afMaskCent[0] = 0.0f;
		m_afMaskCent[1] = 0.0f;
		m_afMaskSize[0] = 1.0f;
		m_afMaskSize[1] = 1.0f;
		m_bSimpleSum = false;
	}
	~CAppParam(void)
	{
	}
};

//---------------------------------------------------------
class CAvgFFT
{
public:
	CAvgFFT(void);
	~CAvgFFT(void);
	void Setup(int* piTileSize);
	void Add(float* pfTile, bool bClean);
	void Done(void);
	float* GetAmp(bool bClean);
	float* m_pfAmp;
	int m_aiAmpSize[2];
private:
	void mClean(void);
	double* m_pdSum;
	float* m_pfPadBuf;
	int m_iNumTiles;
	int m_aiTileSize[2];
	int m_aiPadSize[2];
};	//CAvgFFT
//---------------------------------------------------------

class CRotAvg
{
public:
	CRotAvg(void);
	~CRotAvg(void);
	void DoIt(float* pfHalfAmp, int* piSize);
	float* GetAvg(bool bClean);
	float* m_pfAvg;
	int m_iSize;
private:
	float mCalcAvg(int iRadius, int iSteps);
	float mInterpolate(float fX, float fY);
	float* m_pfHalfAmp;
	int m_aiAmpSize[2];
};

}}
