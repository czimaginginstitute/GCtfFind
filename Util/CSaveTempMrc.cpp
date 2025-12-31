#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>

using namespace GCTFFind;

CSaveTempMrc::CSaveTempMrc(void)
{
}

CSaveTempMrc::~CSaveTempMrc(void)
{
}

void CSaveTempMrc::SetFile(char* pcMain, char* pcExt)
{
	memset(m_acMrcFile, 0, sizeof(m_acMrcFile));
	if(pcMain == 0L || pcExt == 0L) return;
	//-------------------------------------
	char acBuf[256];
	strcpy(acBuf, pcMain);
	char* pcTok = strtok(acBuf, ".");
	if(pcTok == 0L) return;
	//---------------------
	strcpy(m_acMrcFile, pcTok);
	strcat(m_acMrcFile, pcExt);
	strcat(m_acMrcFile, ".mrc");
}

void CSaveTempMrc::GDoIt(float* gfImg, int* piSize)
{
	int iPixels = piSize[0] * piSize[1];
	size_t tBytes = iPixels * sizeof(float);
	float* pfBuf = new float[iPixels];
	cudaMemcpy(pfBuf, gfImg, tBytes, cudaMemcpyDeviceToHost);
	this->DoIt(pfBuf, Mrc::eMrcFloat, piSize);
	delete[] pfBuf;
}

void CSaveTempMrc::GDoIt(unsigned char* gucImg, int* piSize)
{
	int iPixels = piSize[0] * piSize[1];
	size_t tBytes = iPixels * sizeof(char);
	unsigned char* pucBuf = new unsigned char[iPixels];
	cudaMemcpy(pucBuf, gucImg, tBytes, cudaMemcpyDeviceToHost);
	this->DoIt(pucBuf, Mrc::eMrcUChar, piSize);
	delete[] pucBuf;
}

void CSaveTempMrc::DoIt(void* pvImg, int iMode, int* piSize)
{
	Mrc::CSaveMrc aSaveMrc;
	if(!aSaveMrc.OpenFile(m_acMrcFile)) return;
	//-----------------------------------------
	aSaveMrc.SetMode(Mrc::eMrcFloat);
	aSaveMrc.SetImgSize(piSize, 1, 1, 1.0f);
	aSaveMrc.SetExtHeader(0, 32, 0);
	//------------------------------
	if(iMode == Mrc::eMrcFloat)
	{	aSaveMrc.DoIt(0, pvImg);
		return;
	}
	//-------------
	int iPixels = piSize[0] * piSize[1];
	float* pfBuf = new float[iPixels];
	if(iMode == Mrc::eMrcUChar || iMode == Mrc::eMrcUCharEM)
	{	unsigned char* pucImg = (unsigned char*)pvImg;
		for(int i=0; i<iPixels; i++)
		{	pfBuf[i] = pucImg[i];
		}
		aSaveMrc.DoIt(0, pfBuf);
	}
	else if(iMode == Mrc::eMrcShort)
	{	short* psImg = (short*)pvImg;
		for(int i=0; i<iPixels; i++)
		{	pfBuf[i] = psImg[i];
		}
		aSaveMrc.DoIt(0, pfBuf);
	}
	else if(iMode == Mrc::eMrcUShort)
	{	unsigned short* pusImg = (unsigned short*)pvImg;
		for(int i=0; i<iPixels; i++)
		{	pfBuf[i] = pusImg[i];
		}
		aSaveMrc.DoIt(0, pfBuf);
	}
	else if(iMode == Mrc::eMrcInt)
	{	int* piImg = (int*)pvImg;
		for(int i=0; i<iPixels; i++)
		{	pfBuf[i] = piImg[i];
		}
		aSaveMrc.DoIt(0, pfBuf);
	}
	else if(iMode == Mrc::eMrc4Bits)
	{	unsigned char* pucImg = new unsigned char[iPixels];
		Mrc::C4BitImage::Unpack(pvImg, pucImg, piSize);
		for(int i=0; i<iPixels; i++)
		{	pfBuf[i] = pucImg[i];
		}
		delete[] pucImg;
		aSaveMrc.DoIt(0, pfBuf);
	}
	delete[] pfBuf;
}	

