all: SparseGrid.h geomFeatures.cpp unitTests.cpp
	g++ -shared -fPIC -std=c++11 -Wall geomFeatures.cpp -o geomFeatures.so -larmadillo
	g++ -std=c++11 -Wall unitTests.cpp -o unitTests.exe -larmadillo

lib: SparseGrid.h geomFeatures.cpp
	g++ -shared -fPIC -std=c++11 -Wall geomFeatures.cpp -o geomFeatures.so -larmadillo

test: SparseGrid.h geomFeatures.cpp unitTests.cpp 
	g++ -std=c++11 -Wall unitTests.cpp -o unitTests.exe -larmadillo
