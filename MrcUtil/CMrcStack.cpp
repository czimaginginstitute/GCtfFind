#include "CMrcUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>

using namespace Thonring::MrcUtil;

CMrcStack::CMrcStack(void)
{
	m_ppfImages = 0L;
	m_iNumImgs = 0;
	m_fPixelSize = 1.0f;
	m_fTiltAxis = 0.0f;
	memset(m_aiStart, 0, sizeof(m_aiStart));
	memset(m_aiImgSize, 0, sizeof(m_aiImgSize));
}

CMrcStack::~CMrcStack(void)
{
	this->CleanImages();
}

void CMrcStack::Create(int* piImgSize, int iNumImgs)
{
	this->CleanImages();
	//------------------
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iNumImgs = iNumImgs;
	if(m_iNumImgs <= 0) return;
	//-------------------------
	m_ppfImgs = new float*[m_iNumImgs];
	memset(m_ppfImgs, 0, sizoef(float*) * m_iNumImgs);
}

void CMrcStack::SetHeader(float fPixelSize, float fTiltAxis)
{
	m_fTiltAxis = fTiltAxis;
	m_fPixelSize = fPixelSize;
}

void CMrcStack::CleanImages(void)
{
	if(m_ppfImgs == 0L) return;
	for(int i=0; i<m_iNumImgs; i++)
	{	if(m_ppfImgs[i] == 0L) continue;
		delete[] m_ppfImgs[i];
	}
	delete[] m_ppfImgs;
	m_ppfImgs = 0L;
}

void CMrcStack::SetImgs
( int iImage, void* pvImage,
  int iMode, bool bClean
)
{	if(iMode == Mrc::eMrcFloat &&
	   m_ppfImgs[iImage] == 0L &&
	   bClean == true)
	{	m_ppfImgs[iImage] = (float*)pvImage;
		return;
	}
	//-------------
	int iPixels = m_aiImgSize[0] * m_aiImgSize[1];
	if(m_ppfImgs[iImage] == 0L) 
	{	m_ppfImgs[iImage] = new float[iPixels];
	}
	float* pfImg = m_ppfImgs[iImage];
	//-------------------------------
	if(iMode == Mrc::eMrcFloat)
	{	memcpy(pfImg, pvImage, sizeof(float) * iPixels);
	}
	//------------------------------------------------------
	else if(iMode == Mrc::eMrcUChar ||
	   iMode == Mrc::eMrcUCharEM)
	{	unsigned char* pucImg = (unsigned char*)pvImage;
		for(int i=0; i<iPixels; i++) pfImg[i] = pucImg[i];
	}
	//--------------------------------------------------------
	else if(iMode == Mrc::eMrcShort)
	{	short* psImg = (short*)pvImage;
		for(int i=0; i<iPixels; i++) pfImg[i] = psImg[i];
	}
	//-------------------------------------------------------
	else if(iMode == Mrc::eMrcUShort)
	{	unsigned short* pusImg = (unsigned short*)pvImage;
		for(int i=0; i<iPixels; i++) pfImg[i] = pusImg[i];
	}
	//--------------------------------------------------------
	else if(iMode == Mrc::eMrcInt)
	{	int* piImg = (int*)pvImage;
		for(int i=0; i<iPixels; i++) pfImg[i] = piImg[i];
	}
	//-------------------------------------------------------
	if(bClean) delete[] (char*)pvImage;
}

float* CMrcStack::GetImage(int iImage, bool bClean)
{
	if(m_ppfImgs == 0L) return 0L;
	if(iImage >= m_iNumImages) return 0L;
	float* pfImg =  m_ppfImgs[iImage];
	if(bClean) m_ppfImgs[iImage] = 0L;
	return pcImg;
}

int CMrcStack::GetPixels(void)
{
	int iPixels = m_aiStkSize[0] * m_aiStkSize[1];
	return iPixels;
}

