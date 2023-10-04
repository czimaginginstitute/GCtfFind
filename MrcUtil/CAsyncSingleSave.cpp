#include "CMrcUtilInc.h"
#include <stdio.h>
#include <memory.h>

using namespace GCTFFind;

CAsyncSingleSave::CAsyncSingleSave(void)
{
	m_pfImage = 0L;
	memset(m_acMrcFile, 0, sizeof(m_acMrcFile));
}

CAsyncSingleSave::~CAsyncSingleSave(void)
{
	if(m_pfImage != 0L) delete[] m_pfImage;
	m_pfImage = 0L;
}

void CAsyncSingleSave::DoIt
(	char* pcMrcFile,
	float* pfImage,
	int* piImgSize,
	float fPixelSize,
	bool bClean,
	bool bAsync
)
{	this->WaitForExit(-1.0f);
     //-----------------------
     strcpy(m_acMrcFile, pcMrcFile);
	memcpy(m_aiImgSize, piImgSize, sizeof(m_aiImgSize));
	if(m_pfImage != 0L) delete[] m_pfImage;
	m_pfImage = pfImage;
	m_fPixelSize = fPixelSize;
	m_bClean = bClean;
	if(bAsync) Util_Thread::Start();
	else mSave();
}

void CAsyncSingleSave::ThreadMain(void)
{
	mSave();
}

void CAsyncSingleSave::mSave(void)
{
	if(m_pfImage == 0L) return;
	//-------------------------
	Mrc::CSaveMrc aSaveMrc;
	if(!aSaveMrc.OpenFile(m_acMrcFile)) 
	{	if(m_bClean) delete[] m_pfImage;
		m_pfImage = 0L;
		return;
	}
	//-------------	
	aSaveMrc.SetMode(Mrc::eMrcFloat);
	aSaveMrc.SetImgSize(m_aiImgSize, 1, 1, m_fPixelSize);
	aSaveMrc.SetExtHeader(0, 0, 0);
	aSaveMrc.DoIt(0, m_pfImage);
	//--------------------------
	if(m_bClean) delete[] m_pfImage;
	m_pfImage = 0L;
}

