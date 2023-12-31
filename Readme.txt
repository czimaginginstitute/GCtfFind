Version 1.0.2
-------------
Bug Fix 08-28-2023:
1. FindCTF/CCTFTheory.cpp::66:
   The formulae of computing the phase shift from amplitude contrast misses the
   the square (CTFFind also misses the square). It should be:
   phase = atan( Ca / sqrt(1 - Ca * Ca) ) where Ca is amplitude contrast.
2. FindCTF/GCalcCTF1D.cu::50 and GCalcCTF2D.cu::94: same mistake as above
3. FindCTF/GCalcCTF2D.cu::112: We should pass in fAddPhase instead of fExtPhase


Version 1.0.3
-------------
Bug Fix: 08-31-2023
1. CProcessMain.cpp: forgot to load angle file. Added pLoadAngFile->DoIt()


Version 1.0.4
-------------
1. Add -LogSpect 1 to the command line. It is used to calculate logrithmic
   spectrum.
   This function is intended for denoised the micrographs whose low
   frequency components are much higher than those in the high-freq domain.
2  Changed defocus search range in FindCtf/CFindCtf1D.cpp to [3000, 30000]
   at 1A pixel size.
3. Added Include and Lib directories. Copied the header and library files
   from Projs/Include, Projs/Lib, CuProjs/Include, and CuProjs/Lib into
   these two folders.

Version 1.0.5
-------------
1. FindCtf/CFindSeriesCtfs::mProcessRefin:
   Added fDfRange = 5000 * pixel_size * pixel_size 
