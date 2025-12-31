#include "CMrcUtilInc.h"
#include "../CMainInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>

using namespace GCTFFind;

CLoadImages* CLoadImages::m_pInstance = 0L;

CLoadImages* CLoadImages::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CLoadImages;
	return m_pInstance;
}

void CLoadImages::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CLoadImages::CLoadImages(void)
{
	m_iMode = 0;
	m_iPixels = 0;
}

CLoadImages::~CLoadImages(void)
{
	this->Clean();
}

void CLoadImages::Clean(void)
{
	pthread_mutex_lock(&m_aMutex);
	while(!m_aLoadedQueue.empty())
	{	m_aLoadedQueue.pop();
	}
	pthread_mutex_unlock(&m_aMutex);
}

CCtfPackage* CLoadImages::GetPackage(bool bPop)
{
	CCtfPackage* pPackage = 0L;
	pthread_mutex_lock(&m_aMutex);
	if(!m_aLoadedQueue.empty())
	{	pPackage = m_aLoadedQueue.front();
		if(bPop) m_aLoadedQueue.pop();
	}
	pthread_mutex_unlock(&m_aMutex);
	return pPackage;
}


void CLoadImages::AsyncLoad(void)
{
	this->Start();
}

void CLoadImages::ThreadMain(void)
{
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	if(pInputFolder->IsTomo()) mReadTomo();
	else mReadMultiple();
}

void CLoadImages::mReadTomo(void)
{
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	int iNumPackages = pInputFolder->GetNumPackages();
	int iZeroTilt = pInputFolder->GetZeroTiltIdx();
	if(iZeroTilt < 0) iZeroTilt = iNumPackages / 2;
	//---------------------------------------------
	bool bClean = true;
	CCtfPackage* pPackage = pInputFolder->GetPackage(iZeroTilt, !bClean);
	//-------------------------------------------------------------------
	Mrc::CLoadMrc aLoadMrc;
	aLoadMrc.OpenFile(pPackage->m_acMrcFileName);
	mLoadPackage(&aLoadMrc, iZeroTilt);
	//---------------------------------
	for(int i=0; i<iNumPackages; i++)
	{	if(i == iZeroTilt) continue;
		else mLoadPackage(&aLoadMrc, i);
		printf("...... image %4d loaded, %4d left.\n",
		   i+1, iNumPackages - 1 - i);
	}	
}

void CLoadImages::mReadMultiple(void)
{
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	int iNumPackages = pInputFolder->GetNumPackages();
	//------------------------------------------------
	bool bSuccess = true, bClean = true;
	Mrc::CLoadMrc aLoadMrc;
	//---------------------
	for(int i=0; i<iNumPackages; i++)
	{	CCtfPackage* pPackage = pInputFolder->GetPackage(i, !bClean);
		bSuccess = aLoadMrc.OpenFile(pPackage->m_acMrcFileName);
		 mLoadPackage(&aLoadMrc, i);
	}
}

void CLoadImages::mLoadPackage(void* pvLoadMrc, int iPackage)
{
	bool bClean = true, bTomo = true;
	Mrc::CLoadMrc* pLoadMrc = (Mrc::CLoadMrc*)pvLoadMrc;
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	bTomo = pInputFolder->IsTomo();
	CCtfPackage* pPackage = pInputFolder->GetPackage(iPackage, !bClean);
	//------------------------------------------------------------------
	pPackage->m_iImgIdx = iPackage;
	pPackage->m_aiImgSize[0] = pLoadMrc->m_pLoadMain->GetSizeX();
	pPackage->m_aiImgSize[1] = pLoadMrc->m_pLoadMain->GetSizeY();
	//-----------------------------------------------------------
	m_iMode = pLoadMrc->m_pLoadMain->GetMode();
	m_iPixels = pPackage->m_aiImgSize[0] * pPackage->m_aiImgSize[1];
	//--------------------------------------------------------------
	int iImage = bTomo ? iPackage : 0;
	if(m_iMode == 2)
	{	pPackage->m_pfImage = new float[m_iPixels];
		pLoadMrc->m_pLoadImg->DoIt(iImage, pPackage->m_pfImage);
	}
	else
	{	int iBytes = Mrc::C4BitImage::GetImgBytes(m_iMode, 
	    	   pPackage->m_aiImgSize);
		char* pcBuf = new char[iBytes];
		pLoadMrc->m_pLoadImg->DoIt(iImage, pcBuf);
		pPackage->m_pfImage = mToFloat(pcBuf);
		delete[] pcBuf;
	}
	mPushQueue(pPackage);
}

void CLoadImages::mPushQueue(CCtfPackage* pPackage)
{
	pthread_mutex_lock(&m_aMutex);
	m_aLoadedQueue.push(pPackage);
	pthread_mutex_unlock(&m_aMutex);
}

float* CLoadImages::mToFloat(char* pcBuf)
{
	float* pfImg = new float[m_iPixels];
	//----------------------------------
	if(m_iMode == Mrc::eMrcUChar || m_iMode == Mrc::eMrcUCharEM)
	{	unsigned char* pucBuf = (unsigned char*)pcBuf;
		for(int i=0; i<m_iPixels; i++)
		{	pfImg[i] = pucBuf[i];
		}
		return pfImg;
	}
	//-------------------
	if(m_iMode == Mrc::eMrcShort)
	{	short* psBuf = (short*)pcBuf;
		for(int i=0; i<m_iPixels; i++)
		{	pfImg[i] = psBuf[i];
		}
		return pfImg;
	}
	//-------------------
	if(m_iMode == Mrc::eMrcFloat)
	{	memcpy(pfImg, pcBuf, sizeof(float) * m_iPixels);
		return pfImg;
	}
	//-------------------
	if(m_iMode == Mrc::eMrcUShort)
	{	unsigned short* pusBuf = (unsigned short*)pcBuf;
		for(int i=0; i<m_iPixels; i++)
		{	pfImg[i] = pusBuf[i];
		}
		return pfImg;
	}
	//-------------------
	delete[] pfImg;
	return 0L;
}

