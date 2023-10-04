#include "CMrcUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

CSaveImages::CSaveImages(void)
{
}

CSaveImages::~CSaveImages(void)
{
	m_aSaveMrc.CloseFile();
}

bool CSaveImages::OpenFile(char* pcMrcFile)
{
	bool bSave = m_aSaveMrc.OpenFile(pcMrcFile);
	if(bSave) return true;
	printf("Warning: unable to save MRC file\n"
	   "   MRC file: %s\n\n", pcMrcFile);
	return false;
}

void CSaveImages::Setup(int* piImgSize, int iNumImgs)
{
	memcpy(m_aiImgSize, piImgSize, sizeof(int) * 2);
	m_iNumImgs = iNumImgs;
	m_aSaveMrc.SetMode(Mrc::eMrcFloat);
	m_aSaveMrc.SetImgSize(m_aiImgSize, m_iNumImgs, 1, 1.0f);
}

void CSaveImages::DoIt(int iImage, float* pfImage, bool bGpu) 
{
	if(!bGpu)
	{	m_aSaveMrc.DoIt(iImage, pfImage);
	}
	else
	{	int iSize = m_aiImgSize[0] * m_aiImgSize[1];
		float* pfBuf = new float[iSize];
		cudaMemcpy(pfBuf, pfImage, sizeof(float) * iSize,
		   cudaMemcpyDefault);
		m_aSaveMrc.DoIt(iImage, pfBuf);
		delete[] pfBuf;
	}
}

