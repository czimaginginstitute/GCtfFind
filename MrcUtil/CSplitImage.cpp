#include "CMrcUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>

using namespace Thonring::MrcUtil;

CSplitImage::CSplitImage(void)
{
	m_ppfTiles = 0L;
	m_piTileStart = 0L;
	m_pfTileMean = 0L;
}

CSplitImage::~CSplitImage(void)
{
	this->Clean();
}

void CSplitImage::SetImage(float* pfImg, int* piImgSize)
{
	m_pfImage = pfImg;
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
}

void CSplitImage::SetTileSize(int* piSize, int* piOverlap)
{
	m_aiTileSize[0] = piSize[0];
	m_aiTileSize[1] = piSize[1];
	m_aiOverlap[0] = piOverlap[0];
	m_aiOverlap[1] = piOverlap[1];
}

void CSplitImage::DoFull(void)
{
	this->Clean();
	mCalcGrid();
	//----------
	m_ppfTiles = new float*[m_iNumTiles];
	memset(m_ppfTiles, 0, sizeof(float) * m_iNumTiles);
	//-------------------------------------------------
	for(int i=0; i<m_iNumTiles; i++)
	{	int* piStart = m_piTileStart + 2 * i;
		m_ppfTiles[i] = mExtractTile(piStart);
	}
}

void CSplitImage::DoTiltAxis(float fTiltAxis, int iOffset)
{
	this->Clean();
	mCalcGrid(fTiltAxis, iOffset);
	//----------------------------
	m_ppfTiles = new float*[m_iNumTiles];
        memset(m_ppfTiles, 0, sizeof(float) * m_iNumTiles);
        //-------------------------------------------------
        for(int i=0; i<m_iNumTiles; i++)
        {       int* piStart = m_piTileStart + 2 * i;
                m_ppfTiles[i] = mExtractTile(piStart);
        }
}

float* CSplitImage::GetTile(int iTile, bool bClean)
{
	float* pfTile = m_ppfTiles[iTile];
	if(bClean) m_ppfTiles[iTile] = 0L;
	return pfTile;
}

void CSplitImage::GetTilePos(int iTile, int* piStart)
{
	piStart[0] = m_piTileStart[2 * iTile];
	piStart[1] = m_piTileStart[2 * iTile + 1];
}

void CSplitImage::Clean(void)
{
	if(m_piTileStart != 0L)
	{	delete[] m_piTileStart;
		m_piTileStart = 0L;
	}
	//-------------------------
	if(m_pfTileMean != 0L)
	{	delete[] m_pfTileMean;
		m_pfTileMean = 0L;
	}
	//------------------------
	if(m_ppfTiles == 0L) return;
	for(int i=0; i<m_iNumTiles; i++)
	{	if(m_ppfTiles[i] == 0L) continue;
		delete[] m_ppfTiles[i];
	}
	delete[] m_ppfTiles;
	m_ppfTiles = 0L;
}

void CSplitImage::mCalcGrid(void)
{
	int iGridX = (m_aiImgSize[0] + m_aiOverlap[0])
		/ (m_aiTileSize[0] + m_aiOverlap[0]);
	int iGridY = (m_aiImgSize[1] + m_aiOverlap[1])
		/ (m_aiTileSize[1] + m_aiOverlap[1]);
	m_iNumTiles = iGridX * iGridY;
	m_piTileStart = new int[2 * m_iNumTiles];
	m_pfTileMean = new float[m_iNumTiles];
	//------------------------------------
	m_piTileStart[0] = 0;
	m_piTileStart[1] = 0;
	//-------------------
	for(int i=0; i<m_iNumTiles; i++)
	{	int x = i % iGridX;
		int y = i / iGridX;
		int j0 =  2 * i;
		int j1 = j0 + 1;
		m_piTileStart[j0] = x * (m_aiTileSize[0] - m_aiOverlap[0]);
		m_piTileStart[j1] = y * (m_aiTileSize[1] - m_aiOverlap[1]);
	}
}

void CSplitImage::mCalcGrid(float fTiltAxis, int iOffset)
{
	int iTiles = (m_aiImgSize[1] + m_aiOverlap[1])
		/ (m_aiTileSize[1] + m_aiOverlap[1]);
	m_piTileStart = new int[2 * iTiles];
	m_pfTileMean = new float[iTiles];
	//-------------------------------
	double dRad = 4 * atan(1.0) / 180.0;
	float fCos = (float)cos(dRad * m_fTiltAxis);
	float fSin = (float)sin(dRad * m_fTiltAxis);
	//------------------------------------------
	int iCentX = m_aiImgSize[0] / 2;
	int iCentY = m_aiImgSize[1] / 2;
	int iX = iOffset - m_aiTileSize[0] / 2;
	//---------------------------------------
	int iCount = 0;
	for(int y=0; y<iGridY; y++)
	{	int iY = y * (m_aiTileSize[1] - m_aiOverlap[1]);
		int iStartX = (int)(iX * fCos - y * fSin + iCentX);
		int iStartY = (int)(iX * fSin + y * fCos + iCentY);
		int iEndX = iStartX + m_aiTileSize[0];
		int iEndY = iStartY + m_aiTileSize[1];
		if(iStartX < 0 || iStartY < 0) continue;
		if(iEndX >= m_aiImgSize[0]) continue;
		if(iEndY >= m_aiImgSize[1]) continue;
		//-----------------------------------
		int j0 = 2 * iCount;
		int j1 = j0 + 1;
		m_piTileStart[j0] = iStartX;
		m_piTileStart[j1] = iStartY;
		iCount++;
	}
	m_iNumTiles = iCount;
}

void CSplitImage::mExtractTile(int iTile)
{
	int iStartX = m_piTileStart[2 * i];
	int iStartY = m_piTileStart[2 * i + 1];
	int iPixels = m_aiTileSize[0] * m_aiTileSize[1];
	float* pfTile = new float[iPixels];
	//---------------------------------
	int iOffset =  iStartY * m_aiImgSize[0] + iStartX;
	float* pfSrc = m_pfImage + iOffset;
	int iBytes = sizeof(float) * m_aiTileSize[0];
	//-------------------------------------------
	for(int y=0; y<m_aiTileSize; y++)
	{	float* pfDst = pfTile + y * m_aiTileSize[0];
		memcpy(pfDst, pfSrc + y * m_aiImgSize[0], iBytes);
	}
	m_ppfTiles[iTile] = pfTile;
	//-------------------------
	double dMean = 0;
	for(int i=0; i<iPixels; i++)
	{	dMean += pfTile[i];
	}
	m_pfTileMean[iTile] = (float)(dMean / iPixels);
}

float* CSplitImage::GetTile(int iTile, bool bClean)
{
	float* pfTile = m_ppfTiles[iTile];
	if(bClean) m_ppfTiles[iTile] = 0L;
	return pfTile;
}

void CSplitImage::GetTileStart(int iTile, int* piStart)
{
	piStart[0] = m_piTileStart[2 * iTile];
	piStart[1] = m_piTileStart[2 * iTile + 1];
}

float CSplitImage::GetTileMean(int iTile)
{
	return m_pfTileMean[iTile];
}

