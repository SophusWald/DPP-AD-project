These files contain the futhark code for running the benchmark for our
project.

To benchmark the current Futhark implementation:
make sure cuda and futhark is loaded and run the code with the command "make run" from the makefile.

To benchmark with our PPAD-implementation:
clone the git repo https://github.com/p-adema/futhark-rev-scan.git
and with GHC and cabal installed (see https://github.com/diku-dk/howto/blob/main/servers.md) follow the instructions
here https://futhark.readthedocs.io/en/stable/installation.html for
compiling from source.

Then use "make run" as above.

Note that git repo includes the compiler optimizations for add and
vectorised operations. To test the code without these optimizations,
remove the guards leading to diffScanAdd and diffScanVec in vjpSOAC, 
in the file "src/Futhark/AD/Rev/SOAC.hs" in the Futhark directory.
