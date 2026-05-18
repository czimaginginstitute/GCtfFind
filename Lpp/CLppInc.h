#pragma once
#include "../Util/CUtilInc.h"
#include <cufft.h>
#include <queue>

namespace GCTFFind
{

class GCalcPhase2D
{
public:
	GCalcPhase2D(void);
	~GCalcPhase2D(void);
	void DoHalf
	( cufftComplex* gCmp,
	  float* gfHalfPhase,
	  int* piCmpSize
	);
	void DoFull
	( float* gfHalfPhase,
	  int* piCmpSize,
	  float* gfFullPhase,
	  bool bFullPadded
	);
};

class GNormAmpSpectrum
{
public:
	GNormAmpSpectrum(void);
	~GNormAmpSpectrum(void);
	void Clean(void);
	void SetCmpSize(int* piCmpSize);
	void DoIt(float* gfHalfSpect);
	//---------------------------
	int m_aiCmpSize[2];
private:
	void mApplyRamp(float* gfHalfSpect);
	void mNorm(float* gfHalfSpect);
};

class GFindTwinPeaks
{
public:
	GFindTwinPeaks(void);
	~GFindTwinPeaks(void);
	void Clean(void);
	void SetPadSize(int* piPadSize);
	void SetPixSize(float fPixSize);
	void DoIt(float* gfPadSpect);
	float m_afPeak1[2];
	float m_afPeak2[2];
private:
	void mCalcRotSum(void);
	void mFindAvgPeak(void);
	void mFindTwinPeaks(void);
	void mLocalSearch
	( int iCentX,
	  int iCentY,
	  float fRadius,
	  int* piPeakLoc
	);
	//---------------------------
	CCufft2D* m_pCufft2D;
	float* m_gfHalfSpect;
	float* m_gfRotSum;
	float* m_gfPadSpect;
	//---------------------------
	int m_aiPadSize[2];
	int m_aiCmpSize[2];
	float m_fPixSize;
	int m_aiAvgPeak[2]; // from 90-degree rotation average
};

class GGenAmpProjs
{
public:
	GGenAmpProjs(void);
	~GGenAmpProjs(void);
	void Clean(void);
	void SetSpectSize(int* piSpectSize);
	void SetRotAngle(float fRotDegree);
	void DoIt(float* gfFullSpect);
	void SaveProjs(const char* pcImgName);
	//---------------------------
	int m_aiSpectSize[2];
private:
	void mSaveProj(float* gfProj, int iSize, char* pcFileName);
	//---------------------------
	float* m_gfProjX;
	float* m_gfProjY;
	float m_fCos;
	float m_fSin;
	int m_iFringeWidth;
};

class CCalcAmpSpect
{
public:
	CCalcAmpSpect(void);
	~CCalcAmpSpect(void);
	void Clean(void);
	void Setup(int* piImgSize, float fPixSize);
	void DoIt(float* pfImage);
	//---------------------------
	int m_aiPadSize[2];
	float* m_gfPadSpect;
private:
	void mPadImage(float* pfImage);
	void mNormImg(void);
	void mRoundEdge(void);
	void mForwardFFT(void);
	void mCalcSpect(void);
	//---------------------------
	float* m_gfPadImg;
	CCufft2D* m_pCufft2D;
	//---------------------------
	int m_aiRawSize[2];
	float m_fPixSize;
	//---------------------------
	int m_aiImgSize[2];
	int m_aiCmpSize[2];
};
	

class CProcessLpp
{
public:
	static CProcessLpp* GetInstance(void);
	static void DeleteInstance(void);
	//---------------------------
	~CProcessLpp(void);
	void Clean(void);
	void Setup(int* piImgSize, float fBinning);
	void DoIt(void* pCtfPackage);
private:
	CProcessLpp(void);
	void mLowpass(void);
	void mSaveFullSpect(void);
	//---------------------------
	CCalcAmpSpect* m_pCalcAmpSpect;
	GFindTwinPeaks* m_pFindTwinPeaks;
	//---------------------------
	void* m_pvCtfPackage;
	float m_fPixSize;
	//---------------------------
	static CProcessLpp* m_pInstance;
};

}
