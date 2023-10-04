#pragma once
#include <Mrcfile/CMrcFileInc.h>
#include <cuda.h>
#include <cufft.h>

namespace GCTFFind
{

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
