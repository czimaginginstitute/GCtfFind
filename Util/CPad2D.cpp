#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>

using namespace GCTFFind;

CPad2D::CPad2D(void)
{
}

CPad2D::~CPad2D(void)
{
}

void CPad2D::Pad(float* pfImg, int* piImgSize, float* pfPad)
{
	int iBytes = piImgSize[0] * sizeof(float);
	int iPadX = (piImgSize[0] / 2 + 1) * 2;
	//---------------------------
	for(int y=0; y<piImgSize[1]; y++)
	{	float* pfSrc = pfImg + y * piImgSize[0];
		float* pfDst = pfPad + y * iPadX;
		cudaMemcpy(pfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

void CPad2D::Unpad(float* pfPad, int* piPadSize, float* pfImg)
{
	int iImageX = (piPadSize[0] / 2 - 1) * 2;
	int iBytes = iImageX * sizeof(float);
	//---------------------------
	for(int y=0; y<piPadSize[1]; y++)
	{	float* pfSrc = pfPad + y * piPadSize[0];
		float* pfDst = pfImg + y * iImageX;
		cudaMemcpy(pfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

void CPad2D::GetPadSize(int* piImgSize, int* piPadSize)
{
	piPadSize[0] = (piImgSize[0] / 2 + 1) * 2;
	piPadSize[1] = piImgSize[1];
}

void CPad2D::GetImgSize(int* piPadSize, int* piImgSize)
{
	piImgSize[0] = (piPadSize[0] / 2 - 1) * 2;
	piImgSize[1] = piPadSize[1];
}

void CPad2D::GetCmpSize(int* piImgSize, int* piCmpSize)
{
	piCmpSize[0] = piImgSize[0] / 2 + 1;
	piCmpSize[1] = piImgSize[1];
}
