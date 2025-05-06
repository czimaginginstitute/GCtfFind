#include "CFindCTFInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace GCTFFind;

//------------------------------
// Debugging code
//------------------------------
static CSaveImages s_aSaveImages;
//static int s_iCount = 0;

CGenAvgSpectrum::CGenAvgSpectrum(void)
{
	m_fOverlap = 0.50f;
	m_gfTileSpect = 0L;
	m_pGCalcMoment2D = new GCalcMoment2D;
}

CGenAvgSpectrum::~CGenAvgSpectrum(void)
{
	this->Clean();
}

void CGenAvgSpectrum::Clean(void)
{
	if(m_gfTileSpect != 0L) 
	{	cudaFree(m_gfTileSpect);
		m_gfTileSpect = 0L;
	}
	m_pGCalcMoment2D->Clean();
}

void CGenAvgSpectrum::SetSizes(int* piImgSize, int iTileSize)
{
	this->Clean();
	//----------------------------
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iTileSize = iTileSize;
	//----------------------------
	m_aiCmpSize[0] = iTileSize / 2 + 1;
	m_aiCmpSize[1] = iTileSize;
	m_aiPadSize[0] = m_aiCmpSize[0] * 2;
	m_aiPadSize[1] = m_aiCmpSize[1];
	//----------------------------------
	m_iOverlap = (int)(m_iTileSize * m_fOverlap);
	m_iOverlap = m_iOverlap / 2 * 2;
	//-------------------------------------------
	int iSize = m_iTileSize - m_iOverlap;
	m_aiNumTiles[0] = (m_aiImgSize[0] - m_iOverlap) / iSize;
	m_aiNumTiles[1] = (m_aiImgSize[1] - m_iOverlap) / iSize;
	//------------------------------------------------------
	m_aiOffset[0] = (m_aiImgSize[0] - m_aiNumTiles[0] * m_iTileSize
	   + (m_aiNumTiles[0] - 1) * m_iOverlap) / 2;
	m_aiOffset[1] = (m_aiImgSize[1] - m_aiNumTiles[1] * m_iTileSize
	   + (m_aiNumTiles[1] - 1) * m_iOverlap) / 2;
	//-------------------------------------------
	if(m_gfTileSpect != 0L) cudaFree(m_gfTileSpect);
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	size_t tBytes = sizeof(float) * iCmpSize * 3;
        cudaMalloc(&m_gfTileSpect, tBytes);
	m_gfPadTile = m_gfTileSpect + iCmpSize;
	//-------------------------------------------
	m_pGCalcMoment2D->SetSize(m_aiPadSize, true);

	//s_aSaveImages.OpenFile("/home/szheng/home1/Temp/TestTiles.mrc");
	//s_aSaveImages.Setup(m_aiPadSize, m_aiNumTiles[0] * m_aiNumTiles[1]);
}

void CGenAvgSpectrum::DoIt
(	float* gfPadImg,
	float* gfAvgSpect,
	bool bLogSpect
)
{	m_gfPadImg = gfPadImg;
	m_gfAvgSpect = gfAvgSpect;
	m_bLogSpect = bLogSpect;
	//---------------------------
	GAddImages aGAddImages;
	int iNumTiles = m_aiNumTiles[0] * m_aiNumTiles[1];
	float fFactor2 = 1.0f / iNumTiles;
	//---------------------------
	cudaMemset(m_gfAvgSpect, 0, sizeof(float) * 
	   m_aiCmpSize[0] * m_aiCmpSize[1]);
	//---------------------------
	for(int i=0; i<iNumTiles; i++)
	{	mCalcTileSpectrum(i);
		aGAddImages.DoIt(m_gfAvgSpect, 1.0f, m_gfTileSpect, 
		   fFactor2, m_gfAvgSpect, m_aiCmpSize);
	}
	/*	
	s_aSaveImages.OpenFile("/home/shawn.zheng/Temp/TestAvg.mrc");
	s_aSaveImages.Setup(m_aiCmpSize, 1);
	s_aSaveImages.DoIt(0, m_gfAvgSpect, true);
	*/
}

void CGenAvgSpectrum::mCalcTileSpectrum(int iTile)
{	
	mExtractPadTile(iTile);
	//---------------------
	float fMean = m_pGCalcMoment2D->DoIt(m_gfPadTile, 1, true);
	GNormalize2D aGNormalize;
	aGNormalize.DoIt(m_gfPadTile, fMean, 1.0f, m_aiPadSize, true);
	//------------------------------------------------------------
	GRoundEdge aGRoundEdge;
	float afCent[] = {m_iTileSize * 0.5f, m_iTileSize * 0.5f};
	float afSize[] = {m_iTileSize * 1.0f, m_iTileSize * 1.0f};
	aGRoundEdge.SetMask(afCent, afSize);
	aGRoundEdge.DoIt(m_gfPadTile, m_aiPadSize);
	//-----------------------------------------
	GCalcSpectrum aGCalcSpectrum;
	aGCalcSpectrum.DoPad(m_gfPadTile, m_gfTileSpect, m_aiPadSize);
}

void CGenAvgSpectrum::mExtractPadTile(int iTile)
{	
	int iTileX = iTile % m_aiNumTiles[0];
	int iTileY = iTile / m_aiNumTiles[0];
	int iStartX = m_aiOffset[0] + iTileX * (m_iTileSize - m_iOverlap);
	int iStartY = m_aiOffset[1] + iTileY * (m_iTileSize - m_iOverlap);
	//---------------------------
	int iPadImgX = (m_aiImgSize[0] / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * m_iTileSize;
	int iOffset = iStartY * iPadImgX + iStartX;
	//---------------------------
	for(int y=0; y<m_iTileSize; y++)
	{	float* pfSrc = m_gfPadImg + y * iPadImgX + iOffset;
		float* gfDst = m_gfPadTile + y * m_aiPadSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	}
}

