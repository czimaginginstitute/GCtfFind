# GCTFFind
GCtfFind is a new application that robustly estimates the contrast transfer function (CTF) of cryoET tilt series and cryoEM micrographs, essential information needed for cryoET subtomogram averaging and cryoEM single-particle reconstruction. It is part of our endeavor to build a fully-automated high-throughput cryoET processing pipeline that enables real-time reconstruction of tomograms in parallel with tomographic data collection. GCtfFind can serve both cryoET and cryoEM communities as a standalone application for fast and reliable CTF estimation. The highly efficient batch processing mode, achieved by overlapping disk IO with computation, automates CTF estimation on large data sets exempting users from writing Linux scripts, a non-trivial effort for many users without programming experience. GCtfFind automatically determines parameter search range based on the magnification of data collection, making it robust and convenient to process data sets collected across a broad range of defocus.

![ReadmeImg](https://github.com/czimaginginstitute/GCtfFind/blob/main/docs/ReadmeImg.
png)

## Installation
GCtfFind is developed on Linux platform equipped with at least one Nvidia GPU card. To compile from the source, follow the steps below:

1.	git clone https://github.com/czimaginginstitute/GCtfFind.git
2.	cd GCtfFind 
3.	make exe -f makefile [CUDAHOME=path/cuda-xx.x]

If the compute capability of GPUs is 5.x, use makefile10 instead. If CUDAHOME is not provided, the default installation path of CUDA given in makefile or makefile10 will be used.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com
/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are ex
pected to uphold this code. Please report unacceptable behavior to [opensource@chanzu
ckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contac
ting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
