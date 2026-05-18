#include "CFindCTFInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

CEmbedCTF::CEmbedCTF(void)
{
}

CEmbedCTF::~CEmbedCTF(void)
{
}

void CEmbedCTF::DoIt
(	float* gfFullSpect,
	int* piSpectSize,
	CCTFTheory* pCTFTheory,
	float* pfResRange
)
{	GCalcMeanStd calcMeanStd; // util
	bool bPadded = true;
	calcMeanStd.DoStd(gfFullSpect, piSpectSize, !bPadded);
	float fMean = calcMeanStd.m_fMean;
	float fStd = calcMeanStd.m_fStd;
	//---------------------------
	float fPixelSize = pCTFTheory->GetPixelSize();
	float fMinFreq = fPixelSize / pfResRange[0];
	float fMaxFreq = 0.45f;
	float fGain = fStd * 1.5f;
	//---------------------------
	int aiCmpSize[] = {piSpectSize[0] / 2 + 1, piSpectSize[1]};
	float* gfCtfBuf = CSimpleFuncs::GAllocFloat(aiCmpSize);
	//---------------------------
	GCalcCTF2D gCalcCtf2D;
	CCTFParam* pCtfParam = pCTFTheory->GetParam(false);
	gCalcCtf2D.DoIt(pCtfParam, gfCtfBuf, aiCmpSize);
	//---------------------------
	gCalcCtf2D.EmbedCtf(gfCtfBuf, fMinFreq, fMaxFreq,
	   fMean, fGain, gfFullSpect, aiCmpSize);
	//---------------------------
	if(gfCtfBuf != 0L) cudaFree(gfCtfBuf);
}
