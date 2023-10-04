#include "CMainInc.h"
#include "FindCTF/CFindCTFInc.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace GCTFFind;

CSaveCtfResults* CSaveCtfResults::m_pInstance = 0L;

CSaveCtfResults* CSaveCtfResults::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CSaveCtfResults;
	return m_pInstance;
}

void CSaveCtfResults::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CSaveCtfResults::CSaveCtfResults(void)
{
}

CSaveCtfResults::~CSaveCtfResults(void)
{
}

void CSaveCtfResults::SaveCTF(void)
{
	bool bClean = true;
	CCtfPackage* pPackage = 0L;
	CInput* pInput = CInput::GetInstance();
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	//-------------------------------------------------------
	if(strlen(pInput->m_acOutCtfFile) == 0) return;
	FILE* pFile = fopen(pInput->m_acOutCtfFile, "w");
	if(pFile == 0L)
	{	printf("CTF file: %s\n", pInput->m_acOutCtfFile);
		printf("   cannot be opened, CTF results not saved.\n\n");
	}
	if(pFile == 0L) return;
	//---------------------
	fprintf(pFile, "# Idx   tilt     dfMin      dfMax   azimuth "
	   "extPhase  score    fileName\n");
	int iNumPackages = pInputFolder->GetNumPackages();	
	for(int i=0; i<iNumPackages; i++)
	{	pPackage = pInputFolder->GetPackage(i, !bClean);
		fprintf(pFile, "%4d  %6.2f %10.2f %10.2f %7.2f " 
		   "%7.2f %8.4f  %s\n", pPackage->m_iImgIdx,
		   pPackage->m_fTilt,  pPackage->m_fDfMin,
		   pPackage->m_fDfMax, pPackage->m_fAzimuth,
		   pPackage->m_fExtPhase, pPackage->m_fScore,
		   pPackage->m_acMrcFileName);
	}
	fclose(pFile);
}

void CSaveCtfResults::SaveImod(void)
{
	bool bClean = true;
	CCtfPackage* pPackage = 0L;
	CInput* pInput = CInput::GetInstance();
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	//-------------------------------------------------------
	if(strlen(pInput->m_acOutCtfFile) == 0) return;
	if(!pInputFolder->IsTomo()) return;
	int iNumPackages = pInputFolder->GetNumPackages();
	//------------------------------------------------
	char acFileName[256];
	strcpy(acFileName, pInput->m_acOutCtfFile);
	char* pcLastDot = strrchr(acFileName, '.');
	if(pcLastDot == 0L) strcat(acFileName, "_Imod.txt");
	else strcpy(pcLastDot, "_Imod.txt");
	//----------------------------------
	FILE* pFile = fopen(acFileName, "w");
	if(pFile == 0L) return;
	//---------------------
	pPackage = pInputFolder->GetPackage(0, !bClean);
	if(pPackage->m_fExtPhase == 0) 
	{	fprintf(pFile, "1  0  0.0  0.0  0.0  3\n");
	}
	else fprintf(pFile, "5  0  0.0  0.0  0.0  3\n");
	//----------------------------------------------
	const char *pcFormat1 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f\n";
	const char *pcFormat2 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f  %8.2f\n";
	//-------------------------------------------------------
	if(pPackage->m_fExtPhase == 0)
	{	for(int i=0; i<iNumPackages; i++)
		{	pPackage = pInputFolder->GetPackage(i, !bClean);
			fprintf(pFile, pcFormat1, i+1, i+1, 
			   pPackage->m_fTilt,  pPackage->m_fTilt,
	   		   pPackage->m_fDfMin * 0.1f, 
			   pPackage->m_fDfMax * 0.1f,
			   pPackage->m_fAzimuth);		   
		}
	}
	else
	{	for(int i=0; i<iNumPackages; i++)
		{	pPackage = pInputFolder->GetPackage(i, !bClean);
			fprintf(pFile, pcFormat2, i+1, i+1, 
			   pPackage->m_fTilt, pPackage->m_fTilt,
			   pPackage->m_fDfMin * 0.1f, 
			   pPackage->m_fDfMax * 0.1f, 
			   pPackage->m_fAzimuth,
			   pPackage->m_fExtPhase);
		}
	}
	fclose(pFile);
}
