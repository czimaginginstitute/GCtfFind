#include "CFindCTFInc.h"
#include <math.h>

using namespace GCTFFind;

float CFindCtfHelp::CalcAstRatio(float fDfMin, float fDfMax)
{
	float fDiff = (fDfMax - fDfMin) * 0.5f;
	float fMean = (fDfMax + fDfMin) * 0.5f;
	//---------------------------
	if(fMean == 0.0f) return 0.0f;
	else return fabs(fDiff / fMean);
}

float CFindCtfHelp::CalcDfMin(float fDfMean, float fAstRatio)
{
	float fDfMin = fDfMean * (1.0f - fAstRatio);
	float fDfMax = fDfMean * (1.0f + fAstRatio);
	if(fabs(fDfMin) < fabs(fDfMax)) return fDfMin;
	else return fDfMax;
}

float CFindCtfHelp::CalcDfMax(float fDfMean, float fAstRatio)
{
	float fDfMin = fDfMean * (1.0f - fAstRatio);
	float fDfMax = fDfMean * (1.0f + fAstRatio);
	if(fabs(fDfMax) > fabs(fDfMin)) return fDfMax;
	else return fDfMin;
}

