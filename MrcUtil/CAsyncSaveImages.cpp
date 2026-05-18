#include "CMrcUtilInc.h"
#include <stdio.h>
#include <memory.h>

using namespace GCTFFind;

CAsyncSaveImages* CAsyncSaveImages::m_pInstance = 0L;

CAsyncSaveImages* CAsyncSaveImages::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CAsyncSaveImages;
	return m_pInstance;
}

void CAsyncSaveImages::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CAsyncSaveImages::CAsyncSaveImages(void)
{
}

CAsyncSaveImages::~CAsyncSaveImages(void)
{
}

void CAsyncSaveImages::AsyncSave(void)
{	
	this->WaitForExit(-1.0f);
	//-----------------------
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	m_iNumPackages = pInputFolder->GetNumPackages();
	if(m_iNumPackages <= 0) return;
	//----------------------------
	this->WaitForExit(-1.0f);
	this->Start();
}

void CAsyncSaveImages::SetPackage(CCtfPackage* pPackage)
{
	pthread_mutex_lock(&m_aMutex);
	m_aSaveQueue.push(pPackage);
	pthread_mutex_unlock(&m_aMutex);
}

void CAsyncSaveImages::ThreadMain(void)
{
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	if(pInputFolder->IsTomo()) mSaveTomo();
	else mSaveMultiple();
}

void CAsyncSaveImages::mSaveTomo(void)
{
	CInput* pInput = CInput::GetInstance();
	CCtfPackage* pPackage = 0L;
	Mrc::CSaveMrc aSaveMrc;
	aSaveMrc.SetMode(Mrc::eMrcFloat);
	//---------------------------
	int iCount = 0;
	while(iCount < m_iNumPackages)
	{	pPackage = mGetPackage();
		if(pPackage == 0L)
		{	this->WaitForExit(0.01f);
			continue;
		}
		//-------------------
		if(iCount == 0)
		{	char acMrcFile[512] = {'\0'};
			pInput->GetOutFile(pPackage->m_acMrcFileName,
		   	   "_CTF.mrc", acMrcFile);
			aSaveMrc.OpenFile(acMrcFile);
			aSaveMrc.SetImgSize(pPackage->m_aiSpectSize,
		   	   m_iNumPackages, 1, 1.0f);
		}
		aSaveMrc.DoIt(pPackage->m_iImgIdx, pPackage->m_pfFullSpect);
		//-------------------
		pPackage->CleanSpects();
		iCount++;
	}
	aSaveMrc.CloseFile();
}

void CAsyncSaveImages::mSaveMultiple(void)
{
	CInput* pInput = CInput::GetInstance();
	CCtfPackage* pPackage = 0L;
	Mrc::CSaveMrc aSaveMrc;
	char acOutMrcFile[512] = {'\0'};
	//---------------------------
	int iCount = 0;
	while(iCount < m_iNumPackages)
	{	pPackage = mGetPackage();
		if(pPackage == 0L)
		{	this->WaitForExit(0.01f);
			continue;
		}
		//-------------------
		pInput->GetOutFile(pPackage->m_acMrcFileName,
		   "_CTF.mrc", acOutMrcFile);
		aSaveMrc.OpenFile(acOutMrcFile);
		aSaveMrc.SetMode(Mrc::eMrcFloat);
		aSaveMrc.SetImgSize(pPackage->m_aiSpectSize, 1, 1, 1.0f);
		aSaveMrc.DoIt(0, pPackage->m_pfFullSpect);
		//-------------------
		pPackage->CleanSpects();
		iCount++;
	}
}

CCtfPackage* CAsyncSaveImages::mGetPackage(void)
{
	CCtfPackage* pPackage = 0L;
	pthread_mutex_lock(&m_aMutex);
	if(!m_aSaveQueue.empty()) 
	{	pPackage = m_aSaveQueue.front();
		m_aSaveQueue.pop();
	}
	pthread_mutex_unlock(&m_aMutex);
	return pPackage;
}
