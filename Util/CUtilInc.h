#pragma once
#include <Mrcfile/CMrcFileInc.h>
#include <cuda.h>
#include <cufft.h>

namespace GCTFFind
{

class CSimpleFuncs
{
public:
	static void CheckCudaError(const char* pcLocation);
	//---------------------------
	static float* GAllocFloat(int* piSize);
	static float* GAllocFloat(int iSize);
	//---------------------------
	static cufftComplex* GAllocCmp(int* piSize);
	static cufftComplex* GAllocCmp(int iSize);
};	// CSimpleFunc

class CParseArgs
{
public:
        CParseArgs(void);
        ~CParseArgs(void);
        void Set(int argc, char* argv[]);
        bool FindVals(const char* pcTag, int aiRange[2]);
        void GetVals(int aiRange[2], float* pfVals);
        void GetVals(int aiRange[2], int* piVal);
        void GetVal(int iArg, char* pcVal);
        void GetVals(int aiRange[2], char** ppcVals);
private:
        char** m_argv;
        int m_argc;
};

class GAddImages
{
public:
	GAddImages(void);
	~GAddImages(void);
	void DoIt
	(  float* gfImage1,
	   float fFactor1,
	   float* gfImage2,
	   float fFactor2,
	   float* gfSum,
	   int* piImgSize
	);
};

class GCalcMeanStd
{
public:
	GCalcMeanStd(void);
	~GCalcMeanStd(void);
	float DoMean(float* gfImg, int* piImgSize, bool bPadded);
	float DoStd(float* pfImg, int* piImgSize, bool bPadded);
	//---------------------------
	float m_fMean;
	float m_fStd;
};

class GCalcCC2D
{
public:
	GCalcCC2D(void);
	~GCalcCC2D(void);
	float DoIt
	( float* gfImg1,
	  float* gfImg2,
	  int* piImgSize,
	  bool bPadded
	);
	float m_fCC;
	float m_afMeanStd1[2];
	float m_afMeanStd2[2];
};

class GCalcMoment2D
{
public:
	GCalcMoment2D(void);
	~GCalcMoment2D(void);
	void Clean(void);
	void SetSize(int* piImgSize, bool bPadded);
	float DoIt(float* gfImg, int iExponent, bool bSync,
	   cudaStream_t stream = 0);
	float GetResult(void);
private:
	void Test(float* gfImg, float fExp);
	int m_iPadX;
	int m_aiImgSize[2];
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
	float* m_gfBuf;

};

class GNormalize2D
{
public:
	GNormalize2D(void);
	~GNormalize2D(void);
	void DoIt(float* gfImg, float fMean, float fStd,
	   int* piImgSize, bool bPadded);
};

class GThreshold2D
{
public:
	GThreshold2D(void);
	~GThreshold2D(void);
	void DoIt(float* gfImg, float fMin, float fMax,
	   int* piImgSize, bool bPadded);
};

class GFtResize2D
{
public:
        GFtResize2D(void);
        ~GFtResize2D(void);
        //-----------------
        static void GetBinnedCmpSize
        (  int* piCmpSize,// cmp size before binning
           float fBin,
           int* piNewSize // cmp size after binning
        );
        static void GetBinnedImgSize
        (  int* piImgSize, // img size before binning
           float fBin,
           int* piNewSize
        );
        static float CalcPixSize
        (  int* piImgSize, // img size before binning
           float fBin,
           float fPixSize  // before binning
        );
        static void GetBinning
        (  int* piCmpSize,  // cmp size before binning
           int* piNewSize,  // cmp size after binning
           float* pfBinning
        );
        void DownSample
        ( cufftComplex* gCmpIn, int* piSizeIn,
          cufftComplex* gCmpOut, int* piSizeOut,
          bool bSum, cudaStream_t stream = 0
        );
        void UpSample
        ( cufftComplex* gCmpIn, int* piSizeIn,
          cufftComplex* gCmpOut, int* piSizeOut,
          cudaStream_t stream = 0
        );
};	// GFtResize2D

class CCufft2D
{
public:
        CCufft2D(void);
        ~CCufft2D(void);
        void CreateForwardPlan(int* piSize, bool bPad);
        void CreateInversePlan(int* piSize, bool bCmp);
        void DestroyPlan(void);
        //-----------------
        bool Forward
        ( float* gfPadImg, cufftComplex* gCmpImg,
          cudaStream_t stream=0
        );
        bool Forward
        ( float* gfPadImg,
          cudaStream_t stream=0
        );
        cufftComplex* ForwardH2G(float* pfImg);
        //-----------------
        bool Inverse
        ( cufftComplex* gCom, float* gfPadImg,
          cudaStream_t stream=0
        );
        bool Inverse
        ( cufftComplex* gCom,
          cudaStream_t stream=0
        );
        float* InverseG2H(cufftComplex* gCmp);
        //-----------------
        void SubtractMean(cufftComplex* gComplex);
private:
        bool mCheckError(cufftResult* pResult, const char* pcFormat);
        const char* mGetErrorEnum(cufftResult error);
        //-----------------
        cufftHandle m_aPlan;
        cufftType m_aType;
        int m_iFFTx;
        int m_iFFTy;
};

class CPad2D
{
public:
        
	CPad2D(void);
	~CPad2D(void);
	void Pad(float* pfImg, int* piImgSize, float* pfPad);
	void Unpad(float* pfPadImg, int* piPadSize, float* pfImg);
	void GetPadSize(int* piImgSize, int* piPadSize);
	void GetImgSize(int* piPadSize, int* piImgSize);
	void GetCmpSize(int* piImgSize, int* piCmpSize);
};

class CRegSpline
{
public:
     CRegSpline(void);
     ~CRegSpline(void);
     float Smooth(float fX);
     void DoIt(float* pfDataX, float* pfDataY, int iSize);
private:
     float mDoIt(float* pfDataX, float* pfDataY, int iSize, int iR);
     void mCalcTerms(float fX, float fR);
     float* m_pfSoln;
     float* m_pfCoeff;
     float* m_pfTerms;
     float m_fR;
     int m_iDim;
};

class CRegSpline2
{
public:
     CRegSpline2(void);
     ~CRegSpline2(void);
     float Smooth(float fX);
	 void SetKnots(float fR1, float fR2);
     float DoIt(float* pfDataX, float* pfDataY, int iSize);
private:
     void mCalcTerms(float fX);
     float* m_pfSoln;
     float* m_pfCoeff;
     float* m_pfTerms;
     float m_fR1;
     float m_fR2;
     int m_iDim;
};

class CRegSpline3
{
public:
    CRegSpline3(void);
    ~CRegSpline3(void);
    float Smooth(float fX);
    void DoIt(float* pfDataX, float* pfDataY, int iSize);
private:
    double mDoIt(float* pfDataX, float* pfDataY, int iSize);
    void mCalcTerms(float fX);
    float* m_pfSoln;
    float* m_pfCoeff;
    float* m_pfTerms;
    float m_fR1;
    float m_fR2;
    float m_fR3;
	int m_iDim;
};

class CSaveTempMrc
{
public:
        CSaveTempMrc(void);
        ~CSaveTempMrc(void);
        void SetFile(char* pcMain, char* pcExt);
        void GDoIt(float* gfImg, int* piSize);
        void GDoIt(unsigned char* gucImg, int* piSize);
        void DoIt(void* pvImg, int iMode, int* piSize);
	void DoStack(float** ppfImgs, int* piImgSize, int iNumImgs);
private:
        char m_acMrcFile[256];
};	//CSaveTempMrc

class CCudaHelper
{
public:
	CCudaHelper(void) {}
	~CCudaHelper(void) {}
	void CheckError(const char* pcLabel);
};

}
