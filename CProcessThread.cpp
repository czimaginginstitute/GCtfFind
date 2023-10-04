#include "CMainInc.h"
#include "FindCTF/CFindCTFInc.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace GCTFFind;

CProcessThread::CProcessThread(void)
{
}

CProcessThread::~CProcessThread(void)
{
}

bool CProcessThread::DoIt(int iNthFile)
{
	bool bExit = this->WaitForExit(1000.0f);
	if(!bExit) return false;
	//----------------------
	m_iNthFile = iNthFile;
	this->Start();
	return true;
}

void CProcessThread::ThreadMain(void)
{
	mSetDevice();
	//----------
	//CTiltSeriesCtf tiltSeriesCtf;
	//tiltSeriesCtf.DoIt();
	//-------------------
	//CCtfResults* pCtfResults = CCtfResults::GetInstance();
	//pCtfResults->SaveCTF();
	//pCtfResults->SaveImod();
}

bool CProcessThread::mSetDevice(void)
{
	CInput* pInput = CInput::GetInstance();
	cudaError_t tErr = cudaSetDevice(pInput->m_iGpuID);
	if(tErr == cudaSuccess) return true;
	//----------------------------------
	printf("\n\nError: Fail to set GPU (%d), ", 0);
	if(tErr == cudaErrorInvalidDevice)
	{	printf("invalid cuda device\n\n");
	}
	else if(tErr == cudaErrorDeviceAlreadyInUse)
	{	printf("device (%d) already in use\n\n", 0);
	}
	return false;
}

