#include "CMainInc.h"
#include "MrcUtil/CMrcUtilInc.h"

using namespace GCTFFind;

CProcessMain::CProcessMain(void)
{
}

CProcessMain::~CProcessMain(void)
{
}

bool CProcessMain::DoIt(void)
{
	CInput* pInput = CInput::GetInstance();
	CInputFolder* pInputFolder = CInputFolder::GetInstance();
	CLoadAngFile* pLoadAngFile = CLoadAngFile::GetInstance();
	CLoadImages* pLoadImages = CLoadImages::GetInstance();
	CAsyncSaveImages* pAsyncSaveImages = CAsyncSaveImages::GetInstance();
	CFindSeriesCtfs* pFindSeriesCtfs = CFindSeriesCtfs::GetInstance();
	CSaveCtfResults* pSaveCtfResults = CSaveCtfResults::GetInstance();
	//----------------------------------------------------------------
	pLoadAngFile->DoIt();
	pInputFolder->ReadFiles();
	pLoadImages->AsyncLoad();
	pAsyncSaveImages->AsyncSave();
	pFindSeriesCtfs->DoIt();
	//----------------------
	pAsyncSaveImages->WaitForExit(36000.0f);
	pSaveCtfResults->SaveCTF();
	pSaveCtfResults->SaveImod();
	//--------------------------	
	return true; 
}

