#include "CMainInc.h"
#include "Util/CUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace GCTFFind;

CSaveCtfFile* CSaveCtfFile::m_pInstance = 0L;

CSaveCtfFile* CSaveCtfFile::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CSaveCtfFile;
	return m_pInstance;
}

void CSaveCtfFile::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CSaveCtfFile::CSaveCtfFile(void)
{
}

CSaveCtfFile::~CSaveCtfFile(void)
{
}

void CSaveCtfFile::DoIt(CTiltSeriesCtf* pTiltSeriesCtf, char* pcFileName)
{
	if(pcFileName == 0L || strlen(pcFileName) == 0) return;
	m_pTiltSeriesCtf = pTiltSeriesCtf;
	m_pcFileName = pcFileName;
	//------------------------
	CLoadAngFile* pLoadAngFile = CLoadAngFile::GetInstance();
	if(pLoadAngFile->m_iNumTilts != 0) mSaveYesTilts();
	else mSaveNoTilts();
}

void CSaveCtfFile::mSaveNoTilts(void)
{
	FILE* pFile = fopen(m_pcFileName, "w");
	if(pFile == 0L) return;
	//---------------------
	float afAstig[3] = {0.0f}, fExtPhase, fScore = 0.0f;
	fprintf(pFile, "# Idx  dfMin (A)  dfMax (A)  Azimuth (deg) "
	   "ExtPhase  score\n");
	for(int i=0; i<m_pTiltSeriesCtf->m_iNumTilts; i++)
	{	m_pTiltSeriesCtf->GetAstig(i, afAstig);
		fExtPhase = m_pTiltSeriesCtf->GetExtPhase(i);
		fScore = m_pTiltSeriesCtf->GetScore(i);
		fprintf(pFile, "%4d   %8.2f  %8.2f  %8.2f  %7.2f  %8.4f\n", 
		   i+1, afAstig[0], afAstig[1], afAstig[2], 
		   fExtPhase, fScore);
	}
	fclose(pFile);
}

void CSaveCtfFile::mSaveYesTilts(void)
{
	FILE* pFile = fopen(m_pcFileName, "w");
	if(pFile == 0L) return;
	//---------------------
	CLoadAngFile* pAngFile = CLoadAngFile::GetInstance();
	float afAstig[3] = {0.0f}, fScore = 0.0f;
	fprintf(pFile, "1   0   0.0   0.0   0.0    3\n");
	for(int i=0; i<m_pTiltSeriesCtf->m_iNumTilts; i++)
	{	m_pTiltSeriesCtf->GetAstig(i, afAstig);
		m_pTiltSeriesCtf->GetExtPhase(i);
		fScore = m_pTiltSeriesCtf->GetScore(i);
		//-------------------------------------
		afAstig[0] *= 0.1f;
		afAstig[1] *= 0.1f;
		//-----------------
		int j = i + 1;
		float fTilt = pAngFile->GetAngle(i);
		fprintf(pFile, "%4d  %4d  %7.2f  %7.2f  %8.2f  %8.2f  %7.2f\n",
		   j, j, fTilt, fTilt, afAstig[0], afAstig[1], afAstig[2]);
	}
	fclose(pFile);
}
