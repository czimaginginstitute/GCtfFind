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

Version 1.0.6
-------------
1. Add lowpass filter and apply it to the background removed spectrum.

Version 1.0.7
-------------
Bug Fix:
1. FindCtf/GCalcCTF1D at line 30: should be 0.5f * fw2 * fs2.
   FindCtf/GCalcCTF1D at line 50: added sqrtf.
2. FindCtf/GCalcCTF2D at line 35: should be 0.5f * fW2 * fS2.
   FindCtf/GCalcCTF2D at line 94: added sqrtf.
3. Revised makefile for new environment setting

Version 1.0.8 [04-11-2025]
--------------------------
Bug fix:
1. GCalcCTF2D::mEmbedCtf: the theoretical CTF was squared. Now change back to
   original CTF.
2. GCalcCTF2D.::mEmbedCtf: iY should be (iCmpY - y) % iCmpY when mapping the
   negative x frequency to positive x frequency.
Changes:
1. Ported AreTomo3's implementation of FindCtf in GCtfFind.
2. Reduced the lowpass strength in CFindDefocus2D.cpp from 100 to 40 to account
   for more high-freq signals, important for phase shift estimation.
3. Added image rescaling: when the pixel size is less 1A, the image is rescaled
   to 1A pixel size by Fourier cropping (CRescaleImage.cpp).

Version 1.1.0 [04-17-2025]
--------------------------
Renamed from version 1.0.8

Version 1.1.1 [05-03-2025]
--------------------------
Bug fix:
1. FindCtf/CFindDefocus1D::mBrutalForceSearch: m_afPhaseRange is [min, max],
   not [center, range].
2. Delete FindCtf/GCalcSpectrum2D.cu, which is not used. The used one is
   GCalcSpectrum.cu.

Version 1.1.2 [05-05-2025]
--------------------------
Bug Fix:
1. FindCtf/CFindCtfBase::Setup2: Removed the if statement, which prevents the
   generation of new averaged spectrum.
2. FindCtf/GCalcCTF1D & GCalcCTF2D: 1D generates CTF and 2D generates CTF^2.
   Now the both generates 1D and 2D CTF.
3. FindCtf/GCalcSpectrum: It generates amplitude spectrum whereas GCC1D and
   GCC2D both correlate with CTF^2.
4. FindCtf/GCC1D: see item 3. Now compare amplitude spectrum with abs(CTF).
5. FindCtf/GCC2D: see item 4. Now compare amplitude spectrum with abs(CTF).
Changes:
1. FindCtf/CFindCtfBase::m_afResRange[1]: set it at 0.8 Nyquist. If it is
   beyond 3.5A, cap it at 3.5A.
2. FindCtf/CFindDefocus2D: reduced the B-factor from 100 to 16 to include
   more high-res Thon rings into correlation.

Version 1.1.3 [05-16-2025]
--------------------------
Bug Fix:
1. FindCtf/GSpectralCC2D: Because of the changes in GCalcCTF2D, we should
   correlate |CTF| - 0.5 with background subtracted amplitude spectrum.
Changes:
1. Added Thon ring resolution output both on screen and into file.
