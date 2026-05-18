Running GCTFFind_1.1.7_Cuda13

1. Load CUDA 13
   ml cuda/13

2. Run GCTFFind_1.1.7
   GCtfFind_1.1.7_04-16-2026 -InMrc ./raw/n26feb09b_ -InSuffix .mrc \
   -PixSize 0.572 -Kv 300 -ExtPhase 90 180 -OutDir ./ver117 \
   -Serial 1 -TileSize 512 -Gpu 0

   1) This command processes all the MRC files in ./raw directory sequentially
      because of -Serial 1.
   2) "-InMrc ./raw/n26feb09b -InSuffix .mrc" means processing n26feb09b_*.mrc
      files and ignores all other files.
   3) "-PixSize 0.572" specifies the pixel size in angstrom of the MRC files.
   4) "-ExtPhase 90 180" specifies the search range for extral phase shift
      due to the phase plate. In this case, the search range centers on 90
      degree with 180-degree range.
   5) "-OutDir ./ver117" specifies the output directory where the results
      are saved. 
   6) "-TileSize 512" specifies square tiles of 512 x 512 pixels for calculating
      the averaged amplitude spectrum.

3. Results:
   1) In the output directory, CTF.txt is generated with the similar contents shown
      below.
   # fileName  tilt  dfMin(A)  dfMax(A) azimuth(d) phase(d)  score  res(A)  pixSize(A) lppPitch1  lppAngle1  lppPitch2  lppAngle2
n26feb09b_00023en2.mrc   0.00     431.2     527.1   -13.9     2.0   0.1907  14.63   0.57  4.18e-03    85.7  4.18e-03    -4.3
n26feb09b_00028en2.mrc   0.00    5698.5    5859.8    38.1     1.0   0.8549   2.55   0.57  5.87e-03     0.4  6.05e-03   -88.6
n26feb09b_00025en2.mrc   0.00    4976.4    5101.9    44.7    28.0   0.8310   2.61   0.57  9.94e-03    79.5  1.01e-02    -9.3
n26feb09b_00031en2.mrc   0.00     579.2     579.2   -30.7   172.0   0.3087  25.60   0.57  4.95e-03    88.9  4.98e-03    -1.0
n26feb09b_00027en2.mrc   0.00    5779.4    5938.9    39.6    50.0   0.8477   2.40   0.57  4.17e-03    85.1  4.18e-03    -4.9
n26feb09b_00024en2.mrc   0.00    4956.3    5082.0    47.7    51.0   0.8476   2.63   0.57  4.15e-03    84.7  4.18e-03    -5.2
n26feb09b_00029en2.mrc   0.00    5697.3    5861.0    39.9    56.0   0.8456   2.50   0.57  4.17e-03    85.2  4.17e-03    -4.8
n26feb09b_00026en2.mrc   0.00    5742.5    5895.8    39.1    52.0   0.8437   2.64   0.57  4.17e-03    85.6  4.17e-03    -3.7
n26feb09b_00030en2.mrc   0.00     715.9     842.4    13.3   165.0   0.2282  25.60   0.57  4.18e-03    85.1  4.18e-03    -4.9
n26feb09b_00022en2.mrc   0.00     528.8     629.5   -19.3   174.0   0.1967  14.63   0.57  4.18e-03    85.9  4.18e-03    -4.1
   
   lppPitch1 is the Fourier spacing in 1/A unit.
   lppAngle1 is the orientation LPP 1 in degree with respect to the x axis.
  
   2) *_CTF.mrc files are generated and prefixed the input MRC file names. They are
      for users to visually inpsect CTF fitting accuracy.
   3) *_AMP.mrc files are used for visual inspection laser fringes and template matching.
