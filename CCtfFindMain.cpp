#include "CMainInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

using namespace GCTFFind;

bool mCheckSame(void);
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
	{	printf("GCtfFind version 1.0.7, Mar 26, 2024\n");
		return 0;
	}
	//------------------------
	pInput->Parse(argc, argv);
	bool bSame = mCheckSame();
	if(bSame) return 1;
	//---------------------------------------------
	bool bSave = mCheckSave(pInput->m_acOutMrcFile);
	bool bGpu = mCheckGPU(pInput->m_iGpuID);
	if(!bSave || !bGpu) return 1;
	//---------------------------
	CProcessMain aProcessMain;
	aProcessMain.DoIt();
	return 0;
}

bool mCheckSame(void)
{
	CInput* pInput = CInput::GetInstance();
	int iSame1, iSame2;
	iSame1 = strcasecmp(pInput->m_acInMrcFile, pInput->m_acOutMrcFile);
	iSame2 = strcasecmp(pInput->m_acInMrcFile, pInput->m_acOutCtfFile);
	if(iSame1 == 0 || iSame2 == 0)
	{	fprintf(stderr, "Error: input and output files have the"
		   "same name,\n\n");
		return true;
	}
	return false;
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
