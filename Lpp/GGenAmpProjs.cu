#include "CLppInc.h"
#include "../CMainInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace GCTFFind;

static __device__ int mGGetInt(float fVal, int iRange)
{
	int iVal = (int)fVal;
	if(iVal < 0) return -1;
	else if(iVal >= iRange) return -1;
	else return iVal;
}


//-----------------------------------------------------------------------------
// 1. Project vertical laser fringes onto x axis. gfFullSpect is the full
//    amplitude spectrum. In the future, it will be the rotated to align
//    fringes to the x and y axes.
// 2. Each block takes care of the projection at one x point.
//-----------------------------------------------------------------------------
static __global__ void mGGenProjX
(	float* gfFullSpect,
	int iSizeX,
	int iSizeY,
	int iFringeWidth,
	float fCos,
	float fSin,
	float* gfProjX
)
{	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= iSizeX) return;
	//---------------------------
	float fSum = 0.0f;
	int iCount = 0;
	float fNewX = x - iSizeX / 2;
	//---------------------------
	for(int i=0; i<iFringeWidth; i++)
	{	float fNewY = i - iFringeWidth / 2;
		float fOldX =  fNewX * fCos + fNewY * fSin + iSizeX / 2;
		float fOldY = -fNewX * fSin + fNewY * fCos + iSizeY / 2;
		int iX = mGGetInt(fOldX, iSizeX);
		int iY = mGGetInt(fOldY, iSizeY);
		if(iX < 0) continue;
		if(iY < 0) continue;
		//-------------------
		float fVal = gfFullSpect[iY * iSizeX + iX];
		fSum += fVal;
		iCount += 1;
	}
	gfProjX[x] = fSum / (iCount + (float)1e-20);
}

static __global__ void mGGenProjY
(	float* gfFullSpect,
	int iSizeX,
	int iSizeY,
	int iFringeWidth,
	float fCos,
	float fSin,
	float* gfProjY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	//---------------------------
	float fSum = 0.0f;
	int iCount = 0;
	float fNewY = y - iSizeY / 2;
	//---------------------------
	for(int i=0; i<iFringeWidth; i++)
	{	float fNewX = i - iFringeWidth / 2;
		float fOldX =  fNewX * fCos + fNewY * fSin + iSizeX / 2;
		float fOldY = -fNewX * fSin + fNewY * fCos + iSizeY / 2;
		int iX = mGGetInt(fOldX, iSizeX);
		int iY = mGGetInt(fOldY, iSizeY);
		if(iX < 0) continue;
		if(iY < 0) continue;
		//-------------------
		float fVal = gfFullSpect[iY * iSizeX + iX];
		fSum += fVal;
		iCount += 1;
	}
	gfProjY[y] = fSum / (iCount + (float)1e-20);
}

GGenAmpProjs::GGenAmpProjs(void)
{
	m_iFringeWidth = 64;
	m_gfProjX = 0L;
	m_gfProjY = 0L;
	memset(m_aiSpectSize, 0, sizeof(int) * 2);
	m_fCos = 1.0f;
	m_fSin = 0.0f;
}

GGenAmpProjs::~GGenAmpProjs(void)
{
	this->Clean();
}

void GGenAmpProjs::Clean(void)
{
	if(m_gfProjX != 0L) cudaFree(m_gfProjX);
	if(m_gfProjY != 0L) cudaFree(m_gfProjY);
	m_gfProjX = 0L;
	m_gfProjY = 0L;
}

void GGenAmpProjs::SetSpectSize(int* piSpectSize)
{
	if(m_aiSpectSize[0] != piSpectSize[0]) this->Clean();
	else if(m_aiSpectSize[1] != piSpectSize[1]) this->Clean();
	memcpy(m_aiSpectSize, piSpectSize, sizeof(int) * 2);
	//---------------------------
	if(m_gfProjX == 0L)
	{	int iBytes = m_aiSpectSize[0] * sizeof(float);
		cudaMalloc(&m_gfProjX, iBytes);
	}
	if(m_gfProjY == 0L)
	{	int iBytes = m_aiSpectSize[1] * sizeof(float);
		cudaMalloc(&m_gfProjY, iBytes);
	}
	//---------------------------
	m_iFringeWidth = m_aiSpectSize[0] * 3 / 10;
}

void GGenAmpProjs::SetRotAngle(float fRotDegree)
{
	float fAngle = fRotDegree * 0.01745329f;
	m_fCos = (float)cos(fAngle);
	m_fSin = (float)sin(fAngle);
}

void GGenAmpProjs::DoIt(float* gfFullSpect)
{
	dim3 aBlockDim(256, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = (m_aiSpectSize[0] + aBlockDim.x - 1) / aBlockDim.x;
	//---------------------------
	mGGenProjX<<<aGridDim, aBlockDim>>>(gfFullSpect, 
	   m_aiSpectSize[0], m_aiSpectSize[1], 
	   m_iFringeWidth, m_fCos, m_fSin, 
	   m_gfProjX);
	//---------------------------
	aBlockDim = dim3(1, 256);
	aGridDim = dim3(1, 1);
	aGridDim.y = (m_aiSpectSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	mGGenProjY<<<aGridDim, aBlockDim>>>(gfFullSpect, 
	   m_aiSpectSize[0], m_aiSpectSize[1],
	   m_iFringeWidth, m_fCos, m_fSin,
	   m_gfProjY);
}

void GGenAmpProjs::SaveProjs(const char* pcImgName)
{
	if(pcImgName == 0L || strlen(pcImgName) == 0) return;
	//--------------------------
	CInput* pInput = CInput::GetInstance();
	char acFileName[512] = {'\0'};
	pInput->GetOutFile(pcImgName, "_ProjX.txt", acFileName);
	mSaveProj(m_gfProjX, m_aiSpectSize[0], acFileName);
	//---------------------------
	pInput->GetOutFile(pcImgName, "_ProjY.txt", acFileName);
	mSaveProj(m_gfProjY, m_aiSpectSize[1], acFileName);
}

void GGenAmpProjs::mSaveProj(float* gfProj, int iSize, char* pcFileName)
{
	float* pfProj = new float[iSize];
	cudaMemcpy(pfProj, gfProj, sizeof(float) * iSize,
	   cudaMemcpyDefault);
	//---------------------------
	FILE* pFile = fopen(pcFileName, "wt");
	printf("Proj: %s\n", pcFileName);
	if(pFile != 0L)
	{	for(int i=0; i<iSize; i++)
		{	fprintf(pFile, "%5d  %.4e\n", i+1, pfProj[i]);
		}
		fclose(pFile);
	}
	if(pfProj != 0L) delete[] pfProj;
}
