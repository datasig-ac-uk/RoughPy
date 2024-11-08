# LibalgebraLite

Libalgebra Lite was originally included as a submodule, but this caused some 
maintenance problems. Since we're getting rid of this module in the medium 
term, we decided to unlink from the main libalgebra lite repository and just 
vendor the code in the main RoughPy repository. This allows us to make 
breaking changes to the libalgebra lite codebase that suite our needs, but 
that we probably don't want to be reflected in the main repository.