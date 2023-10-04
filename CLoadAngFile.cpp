#include "CMainInc.h"
#include "Util/CUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace GCTFFind;

CLoadAngFile* CLoadAngFile::m_pInstance = 0L;

CLoadAngFile* CLoadAngFile::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CLoadAngFile;
	return m_pInstance;
}

void CLoadAngFile::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CLoadAngFile::CLoadAngFile(void)
{
	m_pfTilts = 0L;
	m_iNumTilts = 0;
}

CLoadAngFile::~CLoadAngFile(void)
{
	this->Clean();
}

void CLoadAngFile::Clean(void)
{
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	m_pfTilts = 0L;
	m_iNumTilts = 0;
}

void CLoadAngFile::DoIt(void)
{
	this->Clean();
	//------------
	CInput* pInput = CInput::GetInstance();
	if(strlen(pInput->m_acAngFile) == 0) return;
	FILE* pFile = fopen(pInput->m_acAngFile, "r");
	if(pFile == 0L) return;
	//--------------------------------------------
	int iBufSize = 1024;
	m_pfTilts = new float[iBufSize];
	memset(m_pfTilts, 0, sizeof(float) * iBufSize);
	//---------------------------------------------
	char acLine[256];
	float fTilt = 0.0f;
	while(!feof(pFile))
	{	memset(acLine, 0, sizeof(acLine));
		char* pcRet = fgets(acLine, 256, pFile);
		if(pcRet == 0L) break;
		if(pcRet[0] == '#') continue;
		int iItems = sscanf(acLine, "%f", &fTilt);
		if(iItems == 1 && fabs(fTilt) < 85)
		{	m_pfTilts[m_iNumTilts] = fTilt;
			m_iNumTilts += 1;
		}
		if(m_iNumTilts >= iBufSize) break;
	}
	fclose(pFile);
}

