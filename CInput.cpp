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
	strcpy(m_acOutMrcTag, "-OutMrc");
	strcpy(m_acOutCtfTag, "-OutCtf");
	strcpy(m_acAngFileTag, "-AngFile");
	strcpy(m_acKvTag, "-kV");
	strcpy(m_acCsTag, "-Cs");
	strcpy(m_acAmpContrastTag, "-AmpContrast");
	strcpy(m_acPixelSizeTag, "-PixSize");
	strcpy(m_acExtPhaseTag, "-ExtPhase");
	strcpy(m_acTileSizeTag, "-TileSize");
	strcpy(m_acTiltRangeTag, "-TiltRange");
	strcpy(m_acLogSpectTag, "-LogSpect");
	strcpy(m_acSerialTag, "-Serial");
	strcpy(m_acInSuffixTag, "-InSuffix");
	strcpy(m_acGpuIDTag, "-Gpu");
	//------------------------------------
	m_fKv = 300.0f;
	m_fCs = 2.7f;  // mm
	m_fAmpContrast = 0.07f;
	m_fPixelSize = 1.0f; // A
	m_afExtPhase[0] = 0.0f;  // degree
	m_afExtPhase[1] = 0.0f;  // not search when 0 or negative
	m_afTiltRange[0] = 0.0f; 
	m_afTiltRange[1] = 0.0f;
	m_iLogSpect = 0;
	m_iTileSize = 512;
	m_iGpuID = 0;
	memset(m_acInMrcFile, 0, sizeof(m_acInMrcFile));
	memset(m_acOutMrcFile, 0, sizeof(m_acOutMrcFile));
	memset(m_acOutCtfFile, 0, sizeof(m_acOutCtfFile));
	memset(m_acAngFile, 0, sizeof(m_acAngFile));
	memset(m_acInSuffix, 0, sizeof(m_acInSuffix));
}

CInput::~CInput(void)
{
}

void CInput::ShowTags(void)
{
	printf("%-15s\n"
	  "  1. Input MRC file that contains single image or a stack of\n"
	  "     frames. In the latter case, CTF will be estimated for \n"
	  "     each frame.\n\n", m_acInMrcTag);
	printf("%-15s\n"
	  "  1. Output MRC file that contains the averaged\n"
	  "     amplitude spectrum.\n"
	  "  2. If the input is a stack of frames, the output will be\n"
	  "     a stack of spectra, one for each frame.\n\n", m_acOutMrcTag);
	printf("%-15s\n"
	  "  1. Output text file containing the estimated ctf parameters\n"
	  "     one line per tilt image.\n\n", m_acOutCtfTag);
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
	  "  1. Pixel size in A.\n\n", m_acPixelSizeTag);
	printf("%-15s\n"
	  "  1. Extra phase shift and search range in degree.\n\n", 
	   m_acExtPhaseTag);
	printf("%-15s\n"
          "  1. Tile size in pixels.\n\n", m_acTileSizeTag);
	printf("%-15s\n"
	  "  1. Calculate logrithmic spectrum. It is not enabled by default.\n\n",
	   m_acLogSpectTag);
	printf("%-15s\n"
	  "  1. Image or tilt series with their file names ended with the\n" 
	  "     specified suffix will be loaded for CTF estimation.\n"
	  "  2. In this case, -InMrc and -InSuffix are jointly used to\n"
	  "     to screen files for CTF estimation.\n\n", m_acInSuffixTag);
	printf("%-15s\n"
	  "  1. Enable serial CTS estimation where there are multiple files\n"
	  "     to be processed.\n"
	  "  2. -Serial 1 enables serial processing.\n\n", m_acSerialTag);
	printf("%-15s\n"
          "  1. GPU IDs. Default 0.\n"
          "  2. For multiple GPUs, separate IDs by space.\n"
          "     For example, %s 0 1 2 3 specifies 4 GPUs.\n\n", m_acGpuIDTag);
}

void CInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//------------
	int aiRange[2];
	CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	aParseArgs.FindVals(m_acInMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInMrcFile);
	//-------------------------------------------
	aParseArgs.FindVals(m_acOutMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acOutMrcFile);
	//--------------------------------------------
	aParseArgs.FindVals(m_acOutCtfTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acOutCtfFile);
	//--------------------------------------------
	aParseArgs.FindVals(m_acInSuffixTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInSuffix);
	//------------------------------------------
	aParseArgs.FindVals(m_acAngFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acAngFile);
	//-------------------------------------------
	aParseArgs.FindVals(m_acKvTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fKv);
	//----------------------------------
	aParseArgs.FindVals(m_acCsTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fCs);
	//----------------------------------
	aParseArgs.FindVals(m_acAmpContrastTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fAmpContrast);
	//-------------------------------------------
	aParseArgs.FindVals(m_acPixelSizeTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fPixelSize);
	//-----------------------------------------
	aParseArgs.FindVals(m_acExtPhaseTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afExtPhase);
	if(m_afExtPhase[1] < 0) m_afExtPhase[1] = 0.0f;
	//---------------------------------------------
	aParseArgs.FindVals(m_acTiltRangeTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afTiltRange);
	m_afTiltRange[0] = fmax(m_afTiltRange[0], -70.1f);
	m_afTiltRange[1] = fmin(m_afTiltRange[1], 70.1f);
	//-----------------------------------------------
	aParseArgs.FindVals(m_acTileSizeTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iTileSize);
	//----------------------------------------
	aParseArgs.FindVals(m_acLogSpectTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iLogSpect);
	//----------------------------------------
	aParseArgs.FindVals(m_acSerialTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iSerial);
	//--------------------------------------
	aParseArgs.FindVals(m_acGpuIDTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iGpuID);
	mPrint();
}

void CInput::mPrint(void)
{
	printf("\n");
	printf("%-15s  %s\n", m_acInMrcTag, m_acInMrcFile);
	printf("%-15s  %s\n", m_acInSuffixTag, m_acInSuffix);
	printf("%-15s  %s\n", m_acOutMrcTag, m_acOutMrcFile);
	printf("%-15s  %s\n", m_acOutCtfTag, m_acOutCtfFile);
	printf("%-15s  %s\n", m_acAngFileTag, m_acAngFile);
	//-------------------------------------------------
	printf("%-15s  %f\n", m_acKvTag, m_fKv);
	printf("%-15s  %f\n", m_acCsTag, m_fCs);
	printf("%-15s  %f\n", m_acAmpContrastTag, m_fAmpContrast);
	printf("%-15s  %f\n", m_acPixelSizeTag, m_fPixelSize);
	printf("%-15s  %f  %f\n", m_acExtPhaseTag, 
	   m_afExtPhase[0], m_afExtPhase[1]);
	printf("%-15s  %d\n", m_acTileSizeTag, m_iTileSize);
	printf("%-15s  %.2f  %.2f\n", m_acTiltRangeTag,
	   m_afTiltRange[0], m_afTiltRange[1]);
	printf("%-15s  %d\n", m_acLogSpectTag, m_iLogSpect);
	printf("%-15s  %d\n", m_acSerialTag, m_iSerial);
	printf("%-15s  %d\n", m_acGpuIDTag, m_iGpuID);
	//--------------------------------------------
	printf("\n");
}
