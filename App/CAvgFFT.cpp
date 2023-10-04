#include "CAppInc.h"
#include <Utilfft/CUtilfftInc.h>
#include <memory.h>
#include <stdio.h>

using namespace Thonring::App;

CAvgFFT::CAvgFFT(void)
{
	m_pfAmp = 0L;
	m_pdSum = 0L;
	m_pfPadBuf = 0L;
	m_iNumTiles = 0;
}

CAvgFFT::~CAvgFFT(void)
{
	mClean();
	if(m_pfAmp != 0L) delete[] m_pfAmp;
}

void CAvgFFT::Setup(int* piTileSize)
{
	mClean();
	//-------
	m_aiTileSize[0] = piTileSize;
	m_aiTileSize[1] = piTileSize;
	//---------------------------
	Utilfft::CFFT2D aFFT2D;
	aFFT2D.GetPadSize(m_aiTileSize, m_aiPadSize);
	m_pfPadBuf = aFFT2D.GetPadBuf(m_aiTileSize);
	//------------------------------------------
	m_aiAmpSize[0] = m_aiPadSize[0] / 2;
	m_aiAmpSize[1] = m_aiPadSize[1];
	//------------------------------
	int iAmpSize = m_aiAmpSize[0] * m_aiAmpSize[1];
	m_pfAmp = new float[iAmpSize];
	m_pdSum = new double[iAmpSize];
	memset(m_pdSum, 0, sizeof(double) * iAmpSize);
}

void CAvgFFT::Add(float* pfTile, bool bClean)
{
	Utilfft::CFFT2D aFFT2D;
	aFFT2D.Pad(pfTile, m_aiTileSize, m_pfPadBuf);
	if(bClean && pfTile != 0L) delete[] pfTile;
	//-----------------------------------------
	aFFT2D.Forward(m_pfPadBuf, m_aiPadSize);
	fftwf_complex* pComp = (fftwf_complex*)m_pfPadBuf;
	aFFT2D.CalcHalfAmp(pComp, m_aiAmpSize, m_pfAmp);
	//----------------------------------------------
	int iAmpSize = m_aiAmpSize[0] * m_aiAmpSize[1];
	for(int i=0; i<iAmpSize; i++)
	{	m_pdSum[i] += m_pfAmp[i];
	}
	m_iNumTiles++;
}

void CAvgFFT::Done(void)
{
	if(m_iNumTiles == 0 || m_pfAmp == 0L) return;
	//-------------------------------------------
	int iAmpSize = m_aiAmpSize[0] * m_aiAmpSize[1];
	for(int i=0; i<iAmpSize; i++)
	{	m_pfAmp[i] = (float)(m_pdSum[i] / m_iNumTiles);
	}
	//-----------------------------------------------------
	if(m_pdSum != 0L) delete[] m_pdSum;
	if(m_pfPadBuf != 0L) delete[] m_pfPadBuf;
	m_pdSum = 0L;
	m_pfPadBuf = 0L;
}

float* CAvgFFT::GetAmp(bool bClean)
{
	float* pfAmp = m_pfAmp;
	if(bClean) m_pfAmp = 0L;
	return pfAmp;
}

void CAvgFFT::mClean(void)
{
	if(m_pfAmp != 0L) delete[] m_pfAmp;
        if(m_pdSum != 0L) delete[] m_pdSum;
        if(m_pfPadBuf != 0L) delete[] m_pfPadBuf;
	m_pfAmp = 0L;
        m_pdSum = 0L;
        m_pfPadBuf = 0L;
        m_iNumTiles = 0;
}
