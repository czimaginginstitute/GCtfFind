#include "CMainInc.h"
#include "Util/CUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace GCTFFind;

CInput* CInput::m_pInstance = 0L;

CInput* CInput::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CInput;
	return m_pInstance;
}

void CInput::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CInput::CInput(void)
{
	strcpy(m_acInMrcTag, "-InMrc");
	strcpy(m_acInSuffixTag, "-InSuffix");
	strcpy(m_acInSkipsTag, "-InSkips");
	strcpy(m_acSerialTag, "-Serial");
	//---------------------------
	strcpy(m_acOutDirTag, "-OutDir");
	strcpy(m_acAngFileTag, "-AngFile");
	//---------------------------
	strcpy(m_acKvTag, "-kV");
	strcpy(m_acCsTag, "-Cs");
	strcpy(m_acAmpContrastTag, "-AmpContrast");
	strcpy(m_acPixelSizeTag, "-PixSize");
	//---------------------------
	strcpy(m_acDefocusTag, "-Defocus");
	strcpy(m_acAstRatioTag, "-AstRatio");
	strcpy(m_acAstAngleTag, "-AstAngle");
	strcpy(m_acExtPhaseTag, "-ExtPhase");
	//---------------------------
	strcpy(m_acTileSizeTag, "-TileSize");
	strcpy(m_acTiltRangeTag, "-TiltRange");
	strcpy(m_acLogSpectTag, "-LogSpect");
	strcpy(m_acGpuIDTag, "-Gpu");
	//---------------------------
	m_fKv = 300.0f;
	m_fCs = 2.7f;  // mm
	m_fAmpContrast = 0.07f;
	m_fPixSize = 1.0f; // A
	//---------------------------
	memset(m_afDefocus, 0, sizeof(m_afDefocus));
	m_afAstRatio[0] = 0.05f;  // center value
	m_afAstRatio[1] = 0.1f;  // range
	m_afAstAngle[0] = 0.0f;   // center value
	m_afAstAngle[1] = 180.0f; // range
	memset(m_afExtPhase, 0, sizeof(m_afExtPhase));
	//---------------------------
	memset(m_afTiltRange, 0, sizeof(m_afTiltRange));
	m_iLogSpect = 0;
	m_iTileSize = 512;
	m_iGpuID = 0;
	//---------------------------
	memset(m_acInMrcFile, 0, sizeof(m_acInMrcFile));
	memset(m_acOutDir, 0, sizeof(m_acOutDir));
	memset(m_acAngFile, 0, sizeof(m_acAngFile));
	memset(m_acInSuffix, 0, sizeof(m_acInSuffix));
	memset(m_acInSkips, 0, sizeof(m_acInSkips));
}

CInput::~CInput(void)
{
}

void CInput::ShowTags(void)
{
	printf("%-15s: \n"
	   "  1. Input MRC file that contains single image or a stack of\n"
	   "     frames. In the latter case, CTF will be estimated for \n"
	   "     each frame.\n\n", m_acInMrcTag);
	printf("%-15s\n"
	   "  1. Image or tilt series with their file names ended with the\n"
	   "     specified suffix will be loaded for CTF estimation.\n"
	   "  2. In this case, -InMrc and -InSuffix are jointly used to\n"
	   "     to screen files for CTF estimation.\n\n", m_acInSuffixTag);
	printf("%-15s\n"
	   "  1. Comma separated string tokens used to exclude MRC files\n"
	   "     from being loaded for CTF estimation. If any token is\n"
	   "     found in a MRC file, it will not be loaded.\n"
	   "  2. This input parameter is used only with -Serial 1.\n\n",
	   m_acInSkipsTag);
	printf("%-15s\n"
	   "  1. Enale serial CTS estimation where there are multiple files\n"
	   "     to be processed.\n"
	   "  2. -Serial 1 enables serial processing.\n\n", m_acSerialTag);
	//---------------------------
	printf("%-15s\n"
	   "  1. Output directory that stores the averaged\n"
	   "     amplitude spectrum and CTF results.\n"
	   "  2. If the input is a stack of frames, the output will be\n"
	   "     a stack of spectra, one for each frame.\n\n", m_acOutDirTag);
	printf("%-15s\n"
	   "  1. Input text file that contains a single column for tilt\n"
	   "     angles. The order must match the images in the input\n"
	   "     MRC file.\n"
	   "  2. Optional. When not given, the output CTF file will not\n"
	   "     have columns for tilt angle.\n\n", m_acAngFileTag);
	printf("%-15s\n"
	   "  1. Min and max tilt angles of the tilt series if it is\n"
	   "     collected with a fixed tilt step.\n\n", m_acTiltRangeTag);
	printf("%-15s\n"
	   "  1. High tension in keV.\n\n", m_acKvTag);
	printf("%-15s\n"
	   "  1. Spherical aberration Cs in mm\n\n", m_acCsTag);
	printf("%-15s\n"
	   "  1. Amplitude contrast, default 0.07.\n\n", m_acAmpContrastTag);
	printf("%-15s\n"
	   "  1. Pixel size in A of input micrographs.\n\n", 
	   m_acPixelSizeTag);
	//---------------------------
	printf("%-10s\n"
	   "  1. Central value and range of the defocus to be searched.\n"
	   "  2. The default setting is 0 0, which means that GCTFFind\n"
	   "     determines the search range based on the pixel size.\n"
	   "     The search range is [2000A, 30000A] at 1A pixel size.\n\n",
	   m_acDefocusTag);
	printf("%-10s\n"
	   "  1. Central value and range of the astigmatic ratio to be\n"
	   "     searched. The default setting is 0.05 0.05\n\n",
	   m_acAstRatioTag);
	printf("%-10s\n"
	   "  1. Central value and range of the astigmatic angle to be\n"
	   "     searched. The default value is [0, 180] degrees.\n\n",
	   m_acAstAngleTag);
	printf("%-10s\n"
	   "  1. Central value and range of the extra phase shift to be\n"
	   "     searched.\n"
	   "  2. The default setting is [0, 0] degrees, which means that\n"
	   "     the extra phase shift is 0 degree and not searched.\n\n",
	   m_acExtPhaseTag);
	//---------------------------
	printf("%-15s\n"
	   "  1. Calculate logrithmic spectrum. It is not enabled "
	   "     by default.\n\n", m_acLogSpectTag);
}

void CInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//------------
	int aiRange[2];
	CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	aParseArgs.FindVals(m_acInMrcTag, aiRange, 1);
	aParseArgs.GetVal(aiRange[0], m_acInMrcFile);
	//---------------------------
	aParseArgs.FindVals(m_acOutDirTag, aiRange, 1);
	aParseArgs.GetVal(aiRange[0], m_acOutDir);
	//---------------------------
	aParseArgs.FindVals(m_acInSuffixTag, aiRange, 1);
	aParseArgs.GetVal(aiRange[0], m_acInSuffix);
	//---------------------------
	aParseArgs.FindVals(m_acInSkipsTag, aiRange, 1);
	aParseArgs.GetVal(aiRange[0], m_acInSkips);
	//---------------------------
	aParseArgs.FindVals(m_acAngFileTag, aiRange, 1);
	aParseArgs.GetVal(aiRange[0], m_acAngFile);
	//---------------------------
	aParseArgs.FindVals(m_acKvTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_fKv);
	aParseArgs.FindVals(m_acCsTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_fCs);
	aParseArgs.FindVals(m_acAmpContrastTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_fAmpContrast);
	aParseArgs.FindVals(m_acPixelSizeTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_fPixSize);
	//---------------------------
	aParseArgs.FindVals(m_acDefocusTag, aiRange, 2);
	aParseArgs.GetVals(aiRange, m_afDefocus);
	aParseArgs.FindVals(m_acAstRatioTag, aiRange, 2);
	aParseArgs.GetVals(aiRange, m_afAstRatio);
	aParseArgs.FindVals(m_acAstAngleTag, aiRange, 2);
	aParseArgs.GetVals(aiRange, m_afAstAngle);
	aParseArgs.FindVals(m_acExtPhaseTag, aiRange, 2);
	aParseArgs.GetVals(aiRange, m_afExtPhase);
	//---------------------------
	aParseArgs.FindVals(m_acTiltRangeTag, aiRange, 2);
	aParseArgs.GetVals(aiRange, m_afTiltRange);
	m_afTiltRange[0] = fmax(m_afTiltRange[0], -70.1f);
	m_afTiltRange[1] = fmin(m_afTiltRange[1], 70.1f);
	//---------------------------
	aParseArgs.FindVals(m_acTileSizeTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_iTileSize);
	//---------------------------
	aParseArgs.FindVals(m_acLogSpectTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_iLogSpect);
	//---------------------------
	aParseArgs.FindVals(m_acSerialTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_iSerial);
	//---------------------------
	aParseArgs.FindVals(m_acGpuIDTag, aiRange, 1);
	aParseArgs.GetVals(aiRange, &m_iGpuID);
	//---------------------------
	int iSize = strlen(m_acOutDir);
	if(iSize == 0) strcpy(m_acOutDir, "./");
	else if(m_acOutDir[iSize - 1] != '/') strcat(m_acOutDir, "/");
	//---------------------------
	mPrint();
}

void CInput::mPrint(void)
{
	printf("\n");
	printf("%-15s  %s\n", m_acInMrcTag, m_acInMrcFile);
	printf("%-15s  %s\n", m_acInSuffixTag, m_acInSuffix);
	printf("%-15s  %s\n", m_acInSkipsTag, m_acInSkips);
	printf("%-15s  %s\n", m_acOutDirTag, m_acOutDir);
	printf("%-15s  %s\n", m_acAngFileTag, m_acAngFile);
	//---------------------------
	printf("%-15s  %f\n", m_acKvTag, m_fKv);
	printf("%-15s  %f\n", m_acCsTag, m_fCs);
	printf("%-15s  %f\n", m_acAmpContrastTag, m_fAmpContrast);
	printf("%-15s  %f\n", m_acPixelSizeTag, m_fPixSize);
	//---------------------------
	printf("%-15s  %f  %f\n", m_acDefocusTag, m_afDefocus[0], 
	   m_afDefocus[1]);
	printf("%-15s  %f  %f\n", m_acAstRatioTag, m_afAstRatio[0], 
	   m_afAstRatio[1]);
	printf("%-15s  %f  %f\n", m_acAstAngleTag, m_afAstAngle[0],
	   m_afAstAngle[1]);
	printf("%-15s  %f  %f\n", m_acExtPhaseTag, m_afExtPhase[0], 
	   m_afExtPhase[1]);
	//---------------------------
	printf("%-15s  %d\n", m_acTileSizeTag, m_iTileSize);
	printf("%-15s  %.2f  %.2f\n", m_acTiltRangeTag,
	   m_afTiltRange[0], m_afTiltRange[1]);
	printf("%-15s  %d\n", m_acLogSpectTag, m_iLogSpect);
	printf("%-15s  %d\n", m_acSerialTag, m_iSerial);
	printf("%-15s  %d\n", m_acGpuIDTag, m_iGpuID);
	//---------------------------
	printf("\n");
}

void CInput::GetOutFile
(	const char* pcMrcFile,
	const char* pcSuffix, 
	char* pcOutFile
)
{	char acBuf[256] = {'\0'};
	const char* pcSlash = strrchr(pcMrcFile, '/');
	if(pcSlash == 0L) strcpy(acBuf, pcMrcFile);
	else strcpy(acBuf, &pcSlash[1]);
	//---------------------------
	char* pcMrcToken = strstr(acBuf, ".mrc");
	if(pcMrcToken != 0L) strcpy(pcMrcToken, "");
	//---------------------------
	strcpy(pcOutFile, m_acOutDir);
	strcat(pcOutFile, acBuf);
	//---------------------------
	if(pcSuffix != 0L && strlen(pcSuffix) > 0) 
		strcat(pcOutFile, pcSuffix);

}
