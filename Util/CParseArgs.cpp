#include "CUtilInc.h"
#include <memory.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

using namespace GCTFFind;

CParseArgs::CParseArgs(void)
{
}

CParseArgs::~CParseArgs(void)
{
}

void CParseArgs::Set(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
}

bool CParseArgs::FindVals(const char* pcTag, int aiRange[2])
{
	aiRange[0] = -1;
	aiRange[1] = 0;
	for(int i=1; i<m_argc; i++)
	{	if(strcasecmp(m_argv[i], pcTag) != 0) continue;
		aiRange[0] = i + 1;
		break;
	}
	if(aiRange[0] == -1) return false;
	//--------------------------------
	for(int j=aiRange[0]; j<m_argc; j++)
	{	char* argv = m_argv[j];
		char* endp = 0L;
		double d = strtod(argv, &endp);
		if(argv != endp && *endp == '\0') 
		{	aiRange[1] += 1;
			continue;
		}
		long l = strtol(argv, &endp, 0);
		if(argv != endp && *endp == '\0')
		{	aiRange[1] += 1;
			continue;
		}
		break;
	}
	return true;
}

void CParseArgs::GetVals(int aiRange[2], float* pfVals)
{
	if(aiRange[0] == -1) return;
	else if(aiRange[1] == 0) return;
	//------------------------------
	for(int i=0; i<aiRange[1]; i++)
	{	int j = aiRange[0] + i;
		if(j >= m_argc) return;
		sscanf(m_argv[j], "%f", pfVals + i);
	}
}

void CParseArgs::GetVals(int aiRange[2], int* piVals)
{
	if(aiRange[0] == -1) return;
        else if(aiRange[1] == 0) return;
        //------------------------------
	for(int i=0; i<aiRange[1]; i++)
	{	int j = aiRange[0] + i;
		if(j >= m_argc) return;
		sscanf(m_argv[j], "%d", piVals + i);
	}
}

void CParseArgs::GetVals(int aiRange[2], char** ppcVals)
{
	if(aiRange[0] == -1) return;
        else if(aiRange[1] == 0) return;
        //------------------------------
	for(int i=0; i<aiRange[1]; i++)
	{	int j = aiRange[0] + i;
		if(j >= m_argc) return;
		strcpy(ppcVals[i], m_argv[j]);
	}
}

void CParseArgs::GetVal(int iArg, char* pcVal)
{
	if(iArg <= 0 || iArg >= m_argc) return;
	strcpy(pcVal, m_argv[iArg]);
}
