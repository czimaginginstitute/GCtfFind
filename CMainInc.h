#pragma once 
#include <Util/Util_Thread.h>
#include <stdio.h>
#include <queue>
#include <cufft.h>

namespace GCTFFind
{

class CInput
{
public:
	static CInput* GetInstance(void);
	static void DeleteInstance(void);
	~CInput(void);
	void ShowTags(void);
	void Parse(int argc, char* argv[]);
	char m_acInMrcFile[256];
	char m_acOutMrcFile[256];
	char m_acOutCtfFile[256];
	char m_acAngFile[256];
	char m_acInSuffix[256];
	float m_fKv;
	float m_fCs;
	float m_fAmpContrast;
	float m_fPixelSize;
	float m_afExtPhase[2];
	float m_afTiltRange[2];
	int m_iTileSize;
	int m_iLogSpect;
	int m_iGpuID;
	int m_iSerial;

private:
	CInput(void);
	void mPrint(void);
	int m_argc;
	char** m_argv;
	char m_acInMrcTag[32];
	char m_acOutMrcTag[32];
	char m_acOutCtfTag[32];
	char m_acAngFileTag[32];
	char m_acTiltRangeTag[32];
	char m_acKvTag[32];
	char m_acCsTag[32];
	char m_acAmpContrastTag[32];
	char m_acPixelSizeTag[32];
	char m_acExtPhaseTag[32];
	char m_acTileSizeTag[32];
	char m_acLogSpectTag[32];
	char m_acSerialTag[32];
	char m_acInSuffixTag[32];
	char m_acGpuIDTag[32];
	static CInput* m_pInstance;
};

class CCtfPackage
{
public:
	CCtfPackage(void);
	~CCtfPackage(void);
	void Clean(void);
	void CleanSpects(void);
	char m_acMrcFileName[256];
	int m_iImgIdx; 
	float m_fTilt;
	float* m_pfImage;
	int m_aiImgSize[2];
	float* m_pfHalfSpect;
	float* m_pfFullSpect;
	int m_aiSpectSize[2];
	float m_fDfMin;
	float m_fDfMax;
	float m_fAzimuth;
	float m_fExtPhase;
	float m_fScore;
};

class CInputFolder
{
public:
	static CInputFolder* GetInstance(void);
	static void DeleteInstance(void);
	~CInputFolder(void);
	bool ReadFiles(void);
	char* GetFullPath(int iPackage);  // DONT clean
	bool GetFileName(int iPackage, char* pcFileName);
	bool GetSerial(int iPackage, char* pcSerial);
	CCtfPackage* GetPackage(int iPackage, bool bClean);
	void SetPackage(int iPackage, CCtfPackage* pPackage);
	void DeletePackage(int iPackage);
	int GetNumPackages(void);
	int GetZeroTiltIdx(void);
	int FindZeroTilt(void);
	bool IsTomo(void);
private:
	CInputFolder(void);
	bool mReadSingle(void);
	bool mReadFolder(void);
	bool mOpenDir(void);
	bool mGetDirName(void);
	bool mGetSerial(char* pcFullName, char* pcSerial);
	void mClean(void);
	char m_acDirName[256];
	char m_acPrefix[256];
	char m_acSuffix[256];
	CCtfPackage** m_ppPackages;
	int m_iNumPackages;
	int m_iZeroTilt;
	bool m_bTomo;
	int m_ifd;
	int m_iwd;
	static CInputFolder* m_pInstance;
};

class CSaveCtfResults
{
public:
	static CSaveCtfResults* GetInstance(void);
	static void DeleteInstance(void);
	~CSaveCtfResults(void);
	void SaveCTF(void);
	void SaveImod(void);
private:
	CSaveCtfResults(void);
	static CSaveCtfResults* m_pInstance;
};

class CFindSeriesCtfs
{
public:
	static CFindSeriesCtfs* GetInstance(void);
	static void DeleteInstance(void);
	~CFindSeriesCtfs(void);
	void Clean(void);
	void DoIt(void);
private:
	CFindSeriesCtfs(void);
	void mProcessPackage(int iPackage);
	void mProcessFull(void);
	void mProcessRefine(void);
	void mGetResults(void);
	void mDisplay(void);
	CCtfPackage* m_pRefPackage;
	CCtfPackage* m_pPackage;
	void* m_pvFindCtf2D;
	int m_iNumPackages;
	int m_iNumDone;
	static CFindSeriesCtfs* m_pInstance;
};
	
class CProcessThread : public Util_Thread
{
public:
	CProcessThread(void);
	~CProcessThread(void);
	bool DoIt(int iNthFile);
	void ThreadMain(void); 
private:
	bool mSetDevice(void);
	int m_iNthFile;
};

class CProcessMain
{
public:
	CProcessMain(void);
	~CProcessMain(void);
	bool DoIt(void);
private:
	bool mDoSingle(void);
	bool mDoMultiple(void);
	bool mWaitProcessThread(void);
	bool mWaitLoadThread(void);
	CProcessThread m_aProcessThread;
};

class CLoadAngFile
{
public:
	static CLoadAngFile* GetInstance(void);
	static void DeleteInstance(void);
	~CLoadAngFile(void);
	void Clean(void);
	void DoIt(void);
	int m_iNumTilts;
	float* m_pfTilts;
private:
	CLoadAngFile(void);
	static CLoadAngFile* m_pInstance;
};

class CSaveCtfFile
{
public:
	static CSaveCtfFile* GetInstance(void);
	static void DeleteInstance(void);
	~CSaveCtfFile(void);
	void DoIt(CFindSeriesCtfs* pFindSeriesCtfs);
private:
	CSaveCtfFile(void);
	void mSaveYesTilts(void);
	void mSaveNoTilts(void);
	CFindSeriesCtfs* m_pFindSeriesCtf;
	static CSaveCtfFile* m_pInstance;
};

}
