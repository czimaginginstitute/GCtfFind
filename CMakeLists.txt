cmake_minimum_required(VERSION 3.9)
project(GCTFFind)

set(CMAKE_CXX_STANDARD 11)

add_executable(GCTFFind
        App/CAppInc.h
        App/CAvgFFT.cpp
        App/CRotAvg.cpp
        FindCTF/CCalcBackground.cpp
        FindCTF/CCTFTheory.cpp
        FindCTF/CFindCTFInc.h
        FindCTF/CFindDefocus1D.cpp
        FindCTF/CFindImageCTF.cpp
        FindCTF/CGenAvgSpectrum.cpp
        FindCTF/CRemoveBackground.cpp
        FindCTF/GBackground1D.cpp
        FindCTF/GCalcCTF1D.cpp
        FindCTF/GCalcCTF2D.cpp
        FindCTF/GCalcSpectrum.cpp
        FindCTF/GCC1D.cpp
        FindCTF/GCC2D.cpp
        FindCTF/GRadialAvg.cpp
        FindCTF/GRemoveMean.cpp
        FindCTF/GRmBackground.cpp
        FindCTF/GRoundEdge.cpp
        FindCTF/GScaleToS2.cpp
        MrcUtil/CAsyncSingleSave.cpp
        MrcUtil/CLoadImages.cpp
        MrcUtil/CMrcStack.cpp
        MrcUtil/CMrcUtilInc.h
        MrcUtil/CSplitImage.cpp
        Util/CParseArgs.cpp
        Util/CRegSpline.cpp
        Util/CRegSpline2.cpp
        Util/CRegSpline3.cpp
        Util/CSaveTempMrc.cpp
        Util/CUtilInc.h
        Util/GAddImages.cpp
        CInput.cpp
        CMainInc.h
        CProcessMain.cpp
        CProcessThread.cpp
        GCTFCorMain.cpp)
