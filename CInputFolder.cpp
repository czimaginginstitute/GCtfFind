#include "CMainInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

using namespace GCTFFind;

CInputFolder* CInputFolder::m_pInstance = 0L;

CInputFolder* CInputFolder::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CInputFolder;
	return m_pInstance;
}

void CInputFolder::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CInputFolder::CInputFolder(void)
{
	m_ppPackages = 0L;
	m_iNumPackages = 0;
	m_bTomo = false;
	m_iZeroTilt = -1;
}

CInputFolder::~CInputFolder(void)
{
	this->mClean();
}

bool CInputFolder::IsTomo(void)
{
	return m_bTomo;
}

int CInputFolder::FindZeroTilt(void)
{
	m_iZeroTilt = -1;
	if(!m_bTomo) return m_iZeroTilt;
	//------------------------------
	float fDif = m_ppPackages[0]->m_fTilt - m_ppPackages[1]->m_fTilt;
	if(fabs(fDif) < 0.0001f)
	{	m_iZeroTilt = m_iNumPackages / 2;
		return m_iZeroTilt;
	}
	//-------------------------
	float fMin = 10000.0f;
	for(int i=0; i<m_iNumPackages; i++)
	{	float fTilt = (float)fabs(m_ppPackages[i]->m_fTilt);
		if(fTilt >= fMin) continue;
		fMin = fTilt;
		m_iZeroTilt = i;
	}
	return m_iZeroTilt;
}

int CInputFolder::GetZeroTiltIdx(void)
{
	return m_iZeroTilt;
}

char* CInputFolder::GetFullPath(int iPackage)
{
	if(iPackage >= m_iNumPackages) return 0L;
	return m_ppPackages[iPackage]->m_acMrcFileName;
}

bool CInputFolder::GetFileName(int iPackage, char* pcFileName)
{
	char* pcFullPath = this->GetFullPath(iPackage);
	if(pcFullPath == 0L || strlen(pcFullPath) == 0) return false;
	//-----------------------------------------------------------
	char* pcSlash = strrchr(pcFullPath, '/');
	if(pcSlash == 0L) strcpy(pcFileName, pcFullPath);
	else strcpy(pcFileName, &pcSlash[1]);
	return true;
}

CCtfPackage* CInputFolder::GetPackage(int iPackage, bool bClean)
{
	if(iPackage >= m_iNumPackages) return 0L;
	CCtfPackage* pPackage = m_ppPackages[iPackage];
	if(bClean) m_ppPackages[iPackage] = 0L;
	return pPackage;
}

void CInputFolder::SetPackage(int iPackage, CCtfPackage* pPackage)
{
	if(iPackage >= m_iNumPackages) return;
	if(m_ppPackages[iPackage] != 0L) delete m_ppPackages[iPackage];
	m_ppPackages[iPackage] = pPackage;
}

void CInputFolder::DeletePackage(int iPackage)
{
	if(iPackage >= m_iNumPackages) return;
	if(m_ppPackages[iPackage] == 0L) return;
	delete m_ppPackages[iPackage];
	m_ppPackages[iPackage] = 0L;
}

int CInputFolder::GetNumPackages(void)
{
	return m_iNumPackages;
}

bool CInputFolder::GetSerial(int iPackage, char* pcSerial)
{
	char acFileName[256] = {'\0'};
	bool bSuccess = this->GetFileName(iPackage, acFileName);
	if(!bSuccess) return false;
	//-------------------------
	bSuccess = mGetSerial(acFileName, pcSerial);
	return bSuccess;
}

bool CInputFolder::ReadFiles(void)
{
	this->mClean();
	bool bSuccess = true;
	//-------------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iSerial == 0) 
	{	bSuccess = mReadSingle();
		return bSuccess;
	}
	//----------------------
	bSuccess = mGetDirName();
	if(!bSuccess) return false;
	strcpy(m_acSuffix, pInput->m_acInSuffix);
	printf("Directory: %s\n", m_acDirName);
	printf("Prefix:    %s\n", m_acPrefix);
	printf("Suffix:    %s\n", m_acSuffix);
	//-------------------------------------------------
	// Read all the movies in the specified folder for
	// batch processing.
	//-------------------------------------------------
	if(pInput->m_iSerial >= 1) 
	{	int iNumRead = mReadFolder();
		bSuccess = (iNumRead > 0) ? true : false;
	}
	return bSuccess;
}

bool CInputFolder::mReadSingle(void)
{
	m_bTomo = true;
	CInput* pInput = CInput::GetInstance();
	Mrc::CLoadMrc loadMrc;
	bool bOpen = loadMrc.OpenFile(pInput->m_acInMrcFile);
	if(!bOpen) return false;
	//----------------------
	m_iNumPackages = loadMrc.m_pLoadMain->GetSizeZ();
	m_ppPackages = new CCtfPackage*[m_iNumPackages];
	int iNumFloats = loadMrc.m_pLoadMain->GetNumFloats();
	for(int i=0; i<m_iNumPackages; i++)
	{	CCtfPackage* pPackage = new CCtfPackage;
		strcpy(pPackage->m_acMrcFileName, pInput->m_acInMrcFile);
		pPackage->m_aiImgSize[0] = loadMrc.m_pLoadMain->GetSizeX();
		pPackage->m_aiImgSize[1] = loadMrc.m_pLoadMain->GetSizeY();
		//---------------------------------------------------------
		if(iNumFloats > 1)
		{	loadMrc.m_pLoadExt->DoIt(i);
			loadMrc.m_pLoadExt->GetTilt(&pPackage->m_fTilt, 1);
		}
		else pPackage->m_fTilt = 0.0f;
		//----------------------------
		m_ppPackages[i] = pPackage;
	}
	loadMrc.CloseFile();
	//--------------------------------------------------------
	// If this is a single micrograph, return successfully.
	//--------------------------------------------------------
	if(m_iNumPackages <= 1)
	{	m_bTomo = false;
		m_iZeroTilt = -1;
		return true;
	}
	//--------------------------------------------------------
	// This is a tilt series, assign tilt angle to each image
	// from tilt angle file if exists.
	//--------------------------------------------------------
	m_bTomo = true;
	CLoadAngFile* pLoadAngFile = CLoadAngFile::GetInstance();
	if(pLoadAngFile->m_iNumTilts == m_iNumPackages)
	{	for(int i=0; i<m_iNumPackages; i++)
		{	m_ppPackages[i]->m_fTilt = pLoadAngFile->m_pfTilts[i];
		}
		this->FindZeroTilt();
		return true;
	}
	//---------------------------------------------------
	// This is a tilt series, assign angle to each image
	// based on tilt range.
	//---------------------------------------------------
	float fTiltRange = pInput->m_afTiltRange[1] - 
	   pInput->m_afTiltRange[0];
	if(fabs(fTiltRange) >= 10.0f)
	{	float fTiltStep = fTiltRange / (m_iNumPackages - 1);
		for(int i=0; i<m_iNumPackages; i++)
		{	m_ppPackages[i]->m_fTilt = pInput->m_afTiltRange[0]
			   + i * fTiltStep;
		}
		this->FindZeroTilt();
		return true;
	}
	m_iZeroTilt = -1;
	return true;
}

//--------------------------------------------------------------------
// 1. User passes in a file name that is used as the template to 
//    search for a series stack files containing serial numbers.
// 2. The template file contains the full path that is used to
//    determine the folder containing the series stack files
//--------------------------------------------------------------------
bool CInputFolder::mReadFolder(void)
{
	m_bTomo = false;
	m_iNumPackages = 0;
	//-----------------
	DIR* pDir = opendir(m_acDirName);
	if(pDir == 0L)
	{	fprintf(stderr, "Error: cannot open folder\n   %s"
		   "in CInputFolder::mReadFolder.\n\n", m_acDirName);
		return false;
	}
	//------------------------------------------------------------
	int iPrefix = strlen(m_acPrefix);
	int iSuffix = strlen(m_acSuffix);
	struct dirent* pDirent;
	char *pcPrefix = 0L, *pcSuffix = 0L;
	//----------------------------------
	struct stat statBuf;
	char acFullFile[256] = {'\0'};
	strcpy(acFullFile, m_acDirName);
	char* pcMainFile = acFullFile + strlen(m_acDirName);
	//--------------------------------------------------
	std::queue<CCtfPackage*> loadQueue;
	while(true)
	{	pDirent = readdir(pDir);
		if(pDirent == 0L) break;
		if(pDirent->d_name[0] == '.') continue;
		//-------------------------------------
		if(iPrefix > 0)
		{	pcPrefix = strstr(pDirent->d_name, m_acPrefix);
			if(pcPrefix == 0L) continue;
		}
		//---------------------------------------------------
		// check if the suffix is at the end of the file name
		//---------------------------------------------------
		if(iSuffix > 0)
		{	pcSuffix = strcasestr(pDirent->d_name 
			   + iPrefix, m_acSuffix);
			if(pcSuffix == 0L) continue;
			if(strlen(pcSuffix) != iSuffix) continue;
		}
		//----------------------------------
		// check if this is the latest file
		//----------------------------------
		strcpy(pcMainFile, pDirent->d_name);
		if(stat(acFullFile, &statBuf) < 0) continue;
		//------------------------------------------
		CCtfPackage* pPackage = new CCtfPackage;
		strcpy(pPackage->m_acMrcFileName, acFullFile);
		loadQueue.push(pPackage);
		printf("added: %s\n", pPackage->m_acMrcFileName);
		m_iNumPackages += 1;
	}
	printf("\n");
	closedir(pDir);
	//-------------------------------------------------------
	m_ppPackages = new CCtfPackage*[m_iNumPackages];
	for(int i=0; i<m_iNumPackages; i++)
	{	CCtfPackage* pPackage = loadQueue.front();
		m_ppPackages[i] = pPackage;
		loadQueue.pop();
	}
	return true;
}

bool CInputFolder::mGetSerial(char* pcInFile, char* pcSerial)
{
	char acBuf[256] = {'\0'};
	int iPrefixLen = strlen(m_acPrefix);
	strcpy(acBuf, &pcInFile[iPrefixLen]);
	//-----------------------------------
	int iSuffix = strlen(m_acSuffix);
	if(iSuffix > 0)
	{	char* pcSuffix = strcasestr(acBuf, m_acSuffix);
		if(pcSuffix != 0L) pcSuffix[0] = '\0';
	}
	else
	{	char* pcExt = strcasestr(acBuf, ".mrc");
		if(pcExt != 0L) pcExt[0] = '\0';
	}
	//--------------------------------------
	strcpy(pcSerial, acBuf);
	return true;
}
	
bool CInputFolder::mGetDirName(void)
{
	CInput* pInput = CInput::GetInstance();
	char* pcPrefix = pInput->m_acInMrcFile;
	//-------------------------------------
	char* pcSlash = strrchr(pcPrefix, '/');
	if(pcSlash == 0L)
	{	strcpy(m_acPrefix, pcPrefix);
		strcpy(m_acDirName, "./"); 
		return true;
	}
	//------------------
	memset(m_acDirName, 0, sizeof(m_acDirName));
	int iNumChars = pcSlash - pcPrefix + 1;
	memcpy(m_acDirName, pcPrefix, iNumChars);
	//---------------------------------------
	int iBytes = strlen(pcPrefix) - iNumChars;
	if(iBytes > 0) memcpy(m_acPrefix, pcSlash + 1, iBytes);
	return true;
}

void CInputFolder::mClean(void)
{
	if(m_ppPackages == 0L) return;
	for(int i=0; i<m_iNumPackages; i++)
	{	if(m_ppPackages[i] == 0L) continue;
		delete m_ppPackages[i];
	}
	delete[] m_ppPackages;
	m_iNumPackages = 0;
}

