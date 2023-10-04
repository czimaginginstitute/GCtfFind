#pragma once
#include "../CMainInc.h"
#include <Util/Util_Thread.h>
#include <Mrcfile/CMrcFileInc.h>

namespace GCTFFind
{
class CLoadImages : public Util_Thread
{
public:
	static CLoadImages* GetInstance(void);
	static void DeleteInstance(void);
	~CLoadImages(void);
	void Clean(void);
	CCtfPackage* GetPackage(bool bPop);
	void AsyncLoad(void);
	void ThreadMain(void);
private:
	CLoadImages(void);
	void mReadTomo(void);
	void mReadMultiple(void);
	void mLoadPackage(void* pvLoadMrc, int iPackage);
	void mPushQueue(CCtfPackage* pPackage);
	float* mToFloat(char* pcBuf);
	std::queue<CCtfPackage*> m_aLoadedQueue;
	int m_iMode;
	int m_iPixels;
	static CLoadImages* m_pInstance;
};	//CLoadImages

class CAsyncSaveImages : public Util_Thread
{
public:
	static CAsyncSaveImages* GetInstance(void);
	static void DeleteInstance(void);
	~CAsyncSaveImages(void);
	void AsyncSave(void);
	void SetPackage(CCtfPackage* pPackage);
	void ThreadMain(void);
private:
	CAsyncSaveImages(void);
	void mSaveTomo(void);
	void mSaveMultiple(void);
	void mEmbedSerial(int iPackage, char* pcOutMrcFile);
	CCtfPackage* mGetPackage(void);
	std::queue<CCtfPackage*> m_aSaveQueue;
	int m_iNumPackages;
	static CAsyncSaveImages* m_pInstance;
};


class CSaveImages
{
public:
	CSaveImages(void);
	~CSaveImages(void);
	bool OpenFile(char* pcMrcFile);
	void Setup(int* piImgSize, int iNumImgs);
	void DoIt(int iImage, float* pfImgage, bool bGpu);
private:
	int m_aiImgSize[2];
	int m_iNumImgs;
	Mrc::CSaveMrc m_aSaveMrc;
};

class CAsyncSingleSave : public Util_Thread
{
public:
	CAsyncSingleSave(void);
	~CAsyncSingleSave(void);
	void DoIt
	(  char* pcMrcFile,
	   float* pfImage,
	   int* piImgSize,
	   float fPixelSize,
	   bool bClean,
	   bool bAsync
	);
	void ThreadMain(void);
private:
	void mSave(void);
	char m_acMrcFile[256];
	float* m_pfImage;
	int m_aiImgSize[2];
	float m_fPixelSize;
	bool m_bClean;
};

}
