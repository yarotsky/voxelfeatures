/*
 * unitTests.cpp
 *
 * Unit tests for geomFeatures.cpp
 *
 * Created on: Nov 8, 2016
 * Author: dmitry.yarotsky
 */

#include "geomFeatures.cpp"


void testComputeAngDefectsEdges(){
	printf("testComputeAngDefectsEdges..");
	arma::mat points0;
	points0 << 0 << 0 << 0 << arma::endr
			<< 0 << 0 << 1 << arma::endr
			<< 0 << 1 << 0 << arma::endr
			<< 1 << 0 << 0 << arma::endr
			<< 0 << 1 << 1 << arma::endr;
	arma::mat points = points0*0.5;
	int spatialSize = 200;

	std::vector<std::vector<int>> surfaces;
	std::vector<int> curSurf;
	curSurf = {0, 1, 2};
	surfaces.push_back(curSurf);
	curSurf = {0, 2, 3};
	surfaces.push_back(curSurf);
	curSurf = {3, 2, 1};
	surfaces.push_back(curSurf);
	curSurf = {0, 3, 1};
	surfaces.push_back(curSurf);

	edgeMap defects;
	int exactlyTwoFacesPerEdge;
	exactlyTwoFacesPerEdge = computeAngDefectsEdges(points, surfaces, defects);
	assert(exactlyTwoFacesPerEdge==1);
	std::array<int, 2> edgeBack;
	for (auto it=defects.begin(); it!=defects.end(); it++){
		edgeBack[0] = it->first[1];
		edgeBack[1] = it->first[0];
		assert(it->second == defects.find(edgeBack)->second);
		assert(it->second > 0);
		if ((it->first[0] == 0) || (it->first[1] == 0)) assert(fabs((it->second)/M_PI-0.5)<1e-5);
	}

	SparseGrid grid;
	std::vector<float> features;
	int n, nSpatialSites;
	float sum0, sum1, sum2;

	enum FeatureKind featureKind = EdgeAngularDefect;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	sum0 = 0;
	for (n=0; n<nSpatialSites; n++){
		sum0 += features[n];
	}

    points -= 0.1;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);
	sum1 = 0;
	for (n=0; n<nSpatialSites; n++){
		sum1 += features[n];
	}
    assert(fabs(sum0-sum1) < 1e-5);

    points *= 7;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	sum2 = 0;
	for (n=0; n<nSpatialSites; n++){
		sum2 += features[n];
	}
    assert(fabs(sum2-7*sum1) < 1e-4);

	printf("OK\n");
}


void test_get_features_list_wrap(){
	printf("test_get_features_list_wrap..");
	int n;
	long Nverts = 4;
	long Nfaces = 2;
	double vertA[] = {
			0, 0, 0,
			0, 0, 1,
			0, 1, 0,
			1, 0, 0
	};
	long faceA[] = {
			0, 1, 2,
			0, 1, 3
	};
	for (n=0; n<12; n++) vertA[n] = 0.5*vertA[n]-0.3;
	int spatialSize = 1;
	const char *featureL[] = {"AreaNormal", "Bool", "ScalarArea"};
	const int nFeat = 5;// 3+1+1
	const int NfeatureKinds = sizeof(featureL)/sizeof(featureL[0]);
	int splitEmpty = 1;

	int nSpatialSites;
	long size;
	float *features;
	int *xyz;

	get_features_list_wrap(Nverts, Nfaces, vertA, faceA, spatialSize, &nSpatialSites, &size,
			&features, &xyz, splitEmpty, NfeatureKinds, featureL);
	assert(nSpatialSites == 1);
	assert(size == nFeat);
	assert(fabs(features[3]-1) < 1e-5); // Bool
	assert(fabs(features[4]-0.25) < 1e-5); // ScalarArea

	for (n=0; n<12; n++) vertA[n] = 2*(vertA[n]+0.3)-0.6;
	spatialSize = 2;
	get_features_list_wrap(Nverts, Nfaces, vertA, faceA, spatialSize, &nSpatialSites, &size,
			&features, &xyz, splitEmpty, NfeatureKinds, featureL);
	assert(nSpatialSites == 8);
	assert(size == nSpatialSites*nFeat);
	float sumBool, sumSA;
	sumBool = 0;
	sumSA = 0;
	for (n=0; n<nSpatialSites; n++){
		assert(fabs(features[n*nFeat+3]*(1-features[n*nFeat+3])) < 1e-5); // Bool
		sumBool += features[n*nFeat+3];
		assert(features[n*nFeat+4] > -1e-7);
		sumSA += features[n*nFeat+4];
	}
	assert(fabs(sumSA-1) < 1e-5);

	splitEmpty = 0;
	get_features_list_wrap(Nverts, Nfaces, vertA, faceA, spatialSize, &nSpatialSites, &size,
			&features, &xyz, splitEmpty, NfeatureKinds, featureL);
	assert(nSpatialSites == 4);
	assert(fabs(nSpatialSites-sumBool) < 1e-5);
	assert(size == nSpatialSites*nFeat);
	sumSA = 0;
	for (n=0; n<nSpatialSites; n++){
		assert(fabs(features[n*nFeat+3]-1) < 1e-5);
		assert(features[n*nFeat+4] > -1e-7);
		sumSA += features[n*nFeat+4];
	}
	assert(fabs(sumSA-1) < 1e-5);

	freeme(&features, &xyz);
	printf("OK\n");
}


void testMain(){
	printf("testMain..");
	int n, k;
	arma::mat points0;
	points0 << 0 << 0 << 0 << arma::endr
			<< 0 << 0 << 1 << arma::endr
			<< 0 << 1 << 0 << arma::endr
			<< 1 << 0 << 0 << arma::endr;
	arma::mat points = points0*5-0.1;

	std::vector<std::vector<int>> surfaces;
	std::vector<int> curSurf;
	curSurf = {0, 1, 2};
	surfaces.push_back(curSurf);
	curSurf = {0, 2, 3};
	surfaces.push_back(curSurf);

	SparseGrid grid;
	std::vector<float> features;
	int nSpatialSites, nSpatialSites1;

	int spatialSize = 10;
	enum FeatureKind featureKind = Bool;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] == 1.f);

	spatialSize = 100;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);


	featureKind = ScalarArea;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= 0);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 4);
	int someValueEquals1 = 0;
	for (n=0; n<nSpatialSites; n++){
		if (fabs(features[n]-1) < 1e-5) someValueEquals1 = 1;
	}
	assert(someValueEquals1 == 1);
	float totalArea = 0;
	for (n=0; n<nSpatialSites; n++)	totalArea += features[n];
	assert(fabs(totalArea-25) < 1e-5);

	featureKind = AreaNormal;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -2);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 2);
	totalArea = 0;
	for (n=0; n<nSpatialSites*3; n++)	totalArea += fabs(features[n]);
	assert(fabs(totalArea-25) < 1e-5);

	featureKind = QuadForm;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -4);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 4);
	totalArea = 0;
	for (n=0; n<nSpatialSites; n++)	{
		assert(features[6*n] >= 0);
		assert(features[6*n+1] >= 0);
		assert(features[6*n+2] >= 0);
		assert(features[6*n]*features[6*n+1] >= features[6*n+3]*features[6*n+3]);
		totalArea += fabs(features[6*n]);
		totalArea += fabs(features[6*n+1]);
		totalArea += fabs(features[6*n+2]);
	}
	assert(fabs(totalArea-25) < 1e-5);

	featureKind = EigenValues;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= 0);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 2);
	totalArea = 0;
	for (n=0; n<nSpatialSites*3; n++)	totalArea += fabs(features[n]);
	assert(fabs(totalArea-25) < 1e-5);

	curSurf = {3, 2, 1};
	surfaces.push_back(curSurf);
	curSurf = {0, 3, 1};
	surfaces.push_back(curSurf);

	featureKind = AreaNormal;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	float normalSum[3];
	for (n=0; n<3; n++) normalSum[n] = 0;
	float totalAbsSum = 0;
	for (n=0; n<nSpatialSites; n++)	{
		for (k=0; k<3; k++){
			normalSum[k] += features[3*n+k];
			totalAbsSum += fabs(features[3*n+k]);
		}
	}
	for (k=0; k<3; k++)	assert(fabs(normalSum[k]) < 1e-5);
	assert(totalAbsSum > 50);

	featureKind = VolumeElement;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	float totalVolume = 0;
	for (n=0; n<nSpatialSites; n++)	totalVolume += features[n];
	assert(fabs(fabs(totalVolume)-5*5*5/6.) < 1e-5);

	featureKind = ScalarArea;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);
	totalArea = 0;
	for (n=0; n<nSpatialSites; n++)	totalArea += features[n];

	featureKind = EigenValues;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	float totalAreaEig = 0;
	for (n=0; n<nSpatialSites*3; n++)	totalAreaEig += features[n];
	for (n=0; n<nSpatialSites*3; n++)	assert(features[n] >= -1e-5);
	assert(fabs(totalAreaEig-totalArea) < 1e-5);

	featureKind = EigenValues;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -1e-5);

	// Check Euler characteristic
	double eulerChar;
	eulerChar = getEulerChar(points, surfaces);
	assert(fabs(eulerChar-2) < 1e-5);

	featureKind = VertexAngularDefect;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	double totalDefect = 0;
	for (n=0; n<nSpatialSites; n++)	totalDefect += features[n];
	assert(fabs(eulerChar-totalDefect/(2*M_PI)) < 1e-4);

	featureKind = EdgeAngularDefect;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++){
		assert(features[n] >= 0);
	}

	// many features per voxel
	std::list<enum FeatureKind> featureList = {Bool, AreaNormal, QuadForm, EigenValues,
			VertexAngularDefect, VolumeElement, EdgeAngularDefect};
	assert(nFeaturesPerVoxel_list(featureList) == 16);
	get_features_list(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureList,
			0);
	assert(nSpatialSites*nFeaturesPerVoxel_list(featureList) == (int) features.size());

	printf("OK\n");
}


void testSpatialSize1(){
	printf("testSpatialSize1..");
	int n, k;
	arma::mat points0;
	points0 << 0 << 0 << 0 << arma::endr
			<< 0 << 0 << 1 << arma::endr
			<< 0 << 1 << 0 << arma::endr
			<< 1 << 0 << 0 << arma::endr
			<< 0 << 1 << 1 << arma::endr;
	arma::mat points = points0*0.5-0.1;

	std::vector<std::vector<int>> surfaces;
	std::vector<int> curSurf;
	curSurf = {0, 1, 2};
	surfaces.push_back(curSurf);
	curSurf = {0, 2, 3};
	surfaces.push_back(curSurf);

	SparseGrid grid;
	std::vector<float> features;
	int nSpatialSites, nSpatialSites1;

	int spatialSize = 1;
	enum FeatureKind featureKind = Bool;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] == 1.f);

	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);


	featureKind = ScalarArea;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= 0);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 4);
	float totalArea = 0;
	for (n=0; n<nSpatialSites; n++)	totalArea += features[n];
	assert(fabs(totalArea-0.25) < 1e-5);


	featureKind = AreaNormal;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);

	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -2);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 2);
	totalArea = 0;
	for (n=0; n<nSpatialSites*3; n++)	{
		assert(((fabs(features[n])-0.125) < 1e-5) || (fabs(features[n])) < 1e-5);
	}

	featureKind = QuadForm;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -4);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 4);
	totalArea = 0;
	for (n=0; n<nSpatialSites; n++)	{
		assert(features[6*n] >= 0);
		assert(features[6*n+1] >= 0);
		assert(features[6*n+2] >= 0);
		assert(features[6*n]*features[6*n+1] >= features[6*n+3]*features[6*n+3]);
		totalArea += fabs(features[6*n]);
		totalArea += fabs(features[6*n+1]);
		totalArea += fabs(features[6*n+2]);
	}
	assert(fabs(totalArea-0.25) < 1e-5);


	featureKind = EigenValues;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= 0);
	for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 2);
	totalArea = 0;
	for (n=0; n<nSpatialSites*3; n++)	totalArea += fabs(features[n]);
	assert(fabs(totalArea-0.25) < 1e-5);

	curSurf = {3, 2, 1};
	surfaces.push_back(curSurf);
	curSurf = {0, 3, 1};
	surfaces.push_back(curSurf);


	featureKind = AreaNormal;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	float normalSum[3];
	for (n=0; n<3; n++) normalSum[n] = 0;
	float totalAbsSum = 0;
	for (n=0; n<nSpatialSites; n++)	{
		for (k=0; k<3; k++){
			normalSum[k] += features[3*n+k];
			totalAbsSum += fabs(features[3*n+k]);
		}
	}
	for (k=0; k<3; k++)	assert(fabs(normalSum[k]) < 1e-5);

	featureKind = ScalarArea;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	totalArea = 0;
	for (n=0; n<nSpatialSites; n++)	totalArea += features[n];

	featureKind = EigenValues;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	float totalAreaEig = 0;
	for (n=0; n<nSpatialSites*3; n++)	totalAreaEig += features[n];
	for (n=0; n<nSpatialSites*3; n++)	assert(features[n] >= -1e-5);
	assert(fabs(totalAreaEig-totalArea) < 1e-5);

	featureKind = EigenValues;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -1e-5);


	// Check Euler characteristic
	double eulerChar;
	eulerChar = getEulerChar(points, surfaces);
	assert(fabs(eulerChar-2) < 1e-5);

	featureKind = VertexAngularDefect;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites1,
			spatialSize,
			featureKind,
			0);
	assert(nSpatialSites1 == nSpatialSites);
	assert(nSpatialSites == (int) grid.mp.size());
	assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
	double totalDefect = 0;
	for (n=0; n<nSpatialSites; n++)	totalDefect += features[n];
	assert(fabs(eulerChar-totalDefect/(2*M_PI)) < 1e-4);


	// many features per voxel
	std::list<enum FeatureKind> featureList = {Bool, AreaNormal, QuadForm, EigenValues, VertexAngularDefect};
	assert(nFeaturesPerVoxel_list(featureList) == 14);
	get_features_list(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureList,
			0);
	assert(nSpatialSites*nFeaturesPerVoxel_list(featureList) == (int) features.size());

	surfaces.clear();
	curSurf = {0, 1, 2};
	surfaces.push_back(curSurf);
	curSurf = {1, 2, 4};
	surfaces.push_back(curSurf);

	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	assert(nSpatialSites == 1);

	printf("OK\n");
}


void testEuler(){
	printf("testEuler..");
	int n;
	arma::mat points;
	std::vector<std::vector<int>> surfaces;
	int res;
	const char path[] = "1309429.off";

	res = getMesh(path, points, surfaces);
	if (!(res==0)){
		return;
	}

	double eulerChar;
	eulerChar = getEulerChar(points, surfaces);

	SparseGrid grid;
	std::vector<float> features;
	int nSpatialSites;
	int spatialSize = 10;
	scaleMesh(points, spatialSize);

	enum FeatureKind featureKind = VertexAngularDefect;
	get_features(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureKind,
			0);

	double totalDefect = 0;
	for (n=0; n<nSpatialSites; n++)	totalDefect += features[n];
	assert(fabs(totalDefect/(2*M_PI)-eulerChar) < 1e-4);
	printf("OK\n");
}


void testFullBinaryTree(){
	printf("testFullBinaryTree..");
	int n, k;
	arma::mat points0;
	points0 << 0 << 0 << 0 << arma::endr
			<< 0 << 0 << 1 << arma::endr
			<< 0 << 1 << 0 << arma::endr
			<< 1 << 0 << 0 << arma::endr;

	arma::mat points = 0.3*points0-0.1;

	std::vector<std::vector<int>> surfaces;
	std::vector<int> curSurf;
	curSurf = {0, 1, 2};
	surfaces.push_back(curSurf);
	curSurf = {0, 2, 3};
	surfaces.push_back(curSurf);

	SparseGrid grid;
	std::vector<float> features;
	std::vector<float> featureTotal0;
	std::vector<float> featureTotal1;
	int nSpatialSites0, nSpatialSites1;
	std::list<enum FeatureKind> featureList = {Bool, ScalarArea, AreaNormal,
			QuadForm, EigenValues, VertexAngularDefect, VolumeElement};
	int nFeaturesPV = nFeaturesPerVoxel_list(featureList);
	int spatialSize;
	for (spatialSize=1; spatialSize<20; spatialSize *= 2, points *= 2){
		get_features_list(points,
				surfaces,
				grid,
				features,
				nSpatialSites0,
				spatialSize,
				featureList,
				0);
		assert(nSpatialSites0 == (int) grid.mp.size());
		assert(nSpatialSites0*nFeaturesPV == (int) features.size());
		featureTotal0.clear();
		for (n=0; n<nFeaturesPV; n++){
			float featureTotal = 0;
			for (k=0; k<nSpatialSites0; k++) featureTotal += features[n+k*nFeaturesPV];
			featureTotal0.push_back(featureTotal);
		}
		assert(fabs(featureTotal0[0]-nSpatialSites0) < 1e-5);
		nFeaturesPV = nFeaturesPerVoxel_list(featureList);
		get_features_list(points,
				surfaces,
				grid,
				features,
				nSpatialSites1,
				spatialSize,
				featureList,
				1);
		assert(nSpatialSites1 == (int) grid.mp.size());
		assert(nSpatialSites1*nFeaturesPV == (int) features.size());
		featureTotal1.clear();
		for (n=0; n<nFeaturesPV; n++){
			float featureTotal = 0;
			for (k=0; k<nSpatialSites1; k++) featureTotal += features[n+k*nFeaturesPV];
			featureTotal1.push_back(featureTotal);
		}
		assert(fabs(featureTotal1[0]-nSpatialSites0) < 1e-5);

		for (n=1; n<nFeaturesPV; n++) assert(featureTotal0[n] == featureTotal1[n]);
	}
	printf("OK\n");
}


int main(){
	test_get_features_list_wrap();
	testFullBinaryTree();
	testMain();
	testSpatialSize1();
	testEuler();
	testComputeAngDefectsEdges();
}




