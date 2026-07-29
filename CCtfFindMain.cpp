#include "CMainInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

using namespace GCTFFind;

bool mCheckSave(char* pcMrcFile);
bool mCheckGPU(int iGpuID);

int main(int argc, char* argv[])
{
	CInput* pInput = CInput::GetInstance();
	if(argc == 1)
	{	printf("Use GCtfFind --help to get more information.\n");
		return 0;
	}
	else if(strstr(argv[1], "--help"))
	{	pInput->ShowTags();
		return 0;
	}
	else if(strstr(argv[1], "--version"))
	{	printf("GCtfFind version 1.2.3, Jul 29, 2026\n");
		return 0;
	}
	//---------------------------
	Util_Time utilTime;
	utilTime.Measure();
	//---------------------------
	pInput->Parse(argc, argv);
	bool bGpu = mCheckGPU(pInput->m_iGpuID);
	if(!bGpu) return 1;
	//---------------------------
	CProcessMain aProcessMain;
	aProcessMain.DoIt();
	//---------------------------
	float fSec = utilTime.GetElapsedSeconds();
	printf("Total time:  %.3f (s)\n\n", fSec);
	return 0;
}

bool mCheckSave(char* pcMrcFile)
{
	Mrc::CSaveMrc aSaveMrc;
	bool bSave = aSaveMrc.OpenFile(pcMrcFile);
	remove(pcMrcFile);
	if(bSave) return true;
	//--------------------
	printf("Error: Unable to open output MRC file.\n");
	printf("......%s\n\n", pcMrcFile);
	return false;
}
	
bool mCheckGPU(int iGpuID)
{
	cudaError_t tErr = cudaSetDevice(iGpuID);
	cudaDeviceReset();
	if(tErr == cudaSuccess) return true;
	//----------------------------------
	if(tErr == cudaErrorInvalidDevice)
	{	printf("Error: Invalid GPU (%d)\n\n", iGpuID);
	}
	else if(tErr == cudaErrorDeviceAlreadyInUse)
	{	printf("Error: GPU (%d) already in use\n\n", iGpuID);
	}
	return false;
}
