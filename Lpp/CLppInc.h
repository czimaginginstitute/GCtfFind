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

class GFindTwinPeaks
{
public:
	GFindTwinPeaks(void);
	~GFindTwinPeaks(void);
	void Clean(void);
	void SetSize(int* piCmpSize);
	void DoIt(cufftComplex* gCmp);
private:
	void mCalcPeaks(cufftComplex* gCmp);
	void mFindTwinPeaks(cufftComplex* gCmp);
	//---------------------------
	float* m_gfPeaks;
	int* m_giLocs;
	int m_aiCmpSize[2];
	int m_aiHalfSize[2];
	//---------------------------
	int m_iPeak1;
	int m_iPeak2;
	cufftComplex m_cmpPeak1;
	cufftComplex m_cmpPeak2;
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
	void DoIt(float* pfImage);
	//---------------------------
	void PostProcess(void);
	void SaveSpectrums(void);
private:
	CProcessLpp(void);
	void mPadImage(float* pfImage);
	void mNormImg(float* gfImg, int* piImgSize, bool bPadded);
	void mRoundEdge(float* gfImg, int* piImgSize, bool bPadded);
	void mForwardFFT(void);
	cufftComplex* mFtBinning(void);
	void mCalcAmpPhase(cufftComplex* gCmp);
	void mNormAmp
	( float* gfImg,
	  int* piImgSize,
	  bool bPadded
	);
	void mQueueImg
	( float* gfPadImg,
	  int* piPadSize,
	  bool bAmp
	);
	void mSaveImg
	( float* gfPadImg, 
	  int* piPadSize, 
	  const char* pcSuffix
	);
	void mSaveStack
	( float** ppfImgs,
	  int* piImgSize,
	  int iNumImgs,
	  const char* pcSuffix
	);
	//---------------------------
	int m_aiRawSize[2];
	int m_aiImgSize[2];
	int m_aiImgPadSize[2];
	int m_aiBinSize[2];
	int m_aiBinPadSize[2];
	float* m_gfBuf;
	//---------------------------
	std::queue<float*> m_ampQueue;
	std::queue<float*> m_phiQueue;
	//---------------------------
	float** m_ppfAmps;
	float** m_ppfPhis;
	int m_iNumAmps;
	void mCleanAmps(void);
	void mCorrelateAmps(void);
	//---------------------------

	static CProcessLpp* m_pInstance;
};

}
