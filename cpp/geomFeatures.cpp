/*
 * geomFeatures.cpp
 *
 * A library of routines to generate geometric voxel features for 2D surfaces in 3D
 * defined by collections of triangles (.stl/.off formats).
 *
 * Copyright 2016 Dmitry Yarotsky
 *
 */

#include <armadillo>
#include "SparseGrid.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <list>
#include <ctime>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

enum FeatureKind {
	Bool,
	ScalarArea,
	AreaNormal,
	QuadForm,
	EigenValues,
	VertexAngularDefect,
	EdgeAngularDefect,
	VolumeElement,
	SA = ScalarArea,
	AN = AreaNormal,
	QF = QuadForm,
	EV = EigenValues,
	VAD = VertexAngularDefect,
	EAD = EdgeAngularDefect,
	VE = VolumeElement};

const static struct {
	FeatureKind val;
    const char *str;
} conversion [] = {
    {Bool, "Bool"},
	{ScalarArea, "ScalarArea"},
	{ScalarArea, "SA"},
	{AreaNormal, "AreaNormal"},
	{AreaNormal, "AN"},
	{QuadForm, "QuadForm"},
	{QuadForm, "QF"},
	{EigenValues, "EigenValues"},
	{EigenValues, "EV"},
	{VertexAngularDefect, "VertexAngularDefect"},
	{VertexAngularDefect, "VAD"},
	{EdgeAngularDefect, "EdgeAngularDefect"},
	{EdgeAngularDefect, "EAD"},
	{VolumeElement, "VolumeElement"},
	{VolumeElement, "VE"},
};

FeatureKind str2enum(const char *str)
{
     uint j;
     for (j=0; j<sizeof(conversion)/sizeof(conversion[0]); ++j)
         if (!strcmp (str, conversion[j].str))
             return conversion[j].val;
     printf("Feature %s not recognized\n", str);
     exit(-1);
}

int nFeaturesPerVoxel(enum FeatureKind featureKind){
	int nFeatures = 0;
	if (featureKind == Bool) nFeatures = 1;
	else if (featureKind == ScalarArea) nFeatures = 1;
	else if (featureKind == AreaNormal)	nFeatures = 3;
	else if (featureKind == QuadForm) nFeatures = 6;
	else if (featureKind == EigenValues) nFeatures = 3;
	else if (featureKind == VertexAngularDefect) nFeatures = 1;
	else if (featureKind == EdgeAngularDefect) nFeatures = 1;
	else if (featureKind == VolumeElement) nFeatures = 1;
	return nFeatures;
}

int nFeaturesPerVoxel_list(std::list<enum FeatureKind> featureList){
	int nFeatures = 0;
	for (const auto& featureKind: featureList) nFeatures += nFeaturesPerVoxel(featureKind);
	return nFeatures;
}


static inline void *MallocOrDie(size_t MemSize)
{
    void *AllocMem = malloc(MemSize);
    /* Some implementations return null on a 0 length alloc,
     * we may as well allow this as it increases compatibility
     * with very few side effects */
    if(!AllocMem && MemSize)
    {
        printf("Could not allocate memory!");
        exit(-1);
    }
    return AllocMem;
}


// adapted from boost::hash_combine
 void hash_combine(std::size_t& h, const std::size_t& v)
 {
     h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
 }


class PairHash
{
public:
	std::size_t operator()(const std::array<int, 2ul> &v) const
	{
		size_t h=0;
		int n;
		for (n=0; n<2; n++)	hash_combine(h, std::hash<int>()(v[n]));
		return h;
	}
};

typedef std::unordered_map<std::array<int, 2>, double, PairHash> edgeMap;


struct Node { // voxel on some scale
	int level; // number of binary splits from root node
	double LB[3]; // lower bounds
	double UB[3]; // upper bounds
	struct Node *children; // two children
	struct Node *parent;
	int splitDim; // split dimension producing children (0, 1 or 2)

	long Npoly; // number of mesh faces intersecting the voxel
	double *polygons; // intersections with faces; for each polygon 30 doubles are reserved;
	// by construction, polygons cannot contain more than 9 vertices
	int *polySizes; // number of vertices in polygons
	long *faceInd; // map from polygon to index of original face

	int Nedges; // number of original edges found in the voxel
	double *edges; // intersections of edges with voxels; 6 doubles are reserved for each ([x0, y0, z0, x1, y1, z1])
	int *edgeInds; // point indices ([ind0, ind1])

	int Nverts; // number of original vertices found in the voxel
	long *vertInd; // map from polygon to index of original vertices

};


void destroy_node(struct Node node)
{
	free(node.children);
	free(node.polygons);
	free(node.polySizes);
	free(node.edges);
	free(node.edgeInds);
	free(node.vertInd);
	free(node.faceInd);
}


struct Node initRootNode(long Nverts, long Nfaces, arma::mat const &points, std::vector<std::vector<int>> const &surfaces, double bound){
	struct Node rootNode;
	long n, k, d;
	for (n=0; n<3; n++){
		rootNode.LB[n] = -bound;
		rootNode.UB[n] = bound;
	}

	// faces
	rootNode.Npoly = Nfaces;
	rootNode.polygons = (double*) MallocOrDie(Nfaces*10*3*sizeof(double));
	rootNode.polySizes = (int*) MallocOrDie(Nfaces*sizeof(int));
	rootNode.faceInd = (long*) MallocOrDie(Nfaces*sizeof(long));
	for (n=0; n<Nfaces; n++){
		rootNode.polySizes[n] = 3;
		rootNode.faceInd[n] = n;
		for (k=0; k<3; k++){
			for (d=0; d<3; d++){
				rootNode.polygons[n*10*3+k*3+d] = points(surfaces[n][k],d);
			}
		}
	}

	// edges
	rootNode.Nedges = Nfaces*3;
	rootNode.edges = (double*) MallocOrDie(rootNode.Nedges*6*sizeof(double));
	rootNode.edgeInds = (int*) MallocOrDie(rootNode.Nedges*2*sizeof(int));
	for (n=0; n<Nfaces; n++){
		for (k=0; k<3; k++){
			rootNode.edgeInds[2*(3*n+k)] = surfaces[n][k];
			rootNode.edgeInds[2*(3*n+k)+1] = surfaces[n][(k+1)%3];
			for (d=0; d<3; d++){
				rootNode.edges[6*(3*n+k)+d] = points(surfaces[n][k], d);
				rootNode.edges[6*(3*n+k)+d+3] = points(surfaces[n][(k+1)%3], d);
			}
		}
	}

	// vertices
	rootNode.Nverts = Nverts;
	rootNode.vertInd = (long*) MallocOrDie(Nverts*sizeof(long));
	for (n=0; n<Nverts; n++) rootNode.vertInd[n] = n;

	rootNode.level = 0;
	rootNode.children = NULL;
	assert(rootNode.children == NULL);
	return rootNode;
}


struct Node* splitNode(struct Node *parentPtr, int dim, arma::mat const &points){
	parentPtr->children = (struct Node*) MallocOrDie(2*sizeof(struct Node));
	struct Node *children = parentPtr->children;

	long n, k, k1, d;
	double a;
	double eps = 1e-12;
	parentPtr->splitDim = dim;

	for (n=0; n<2; n++){
		for (d=0; d<3; d++){
			children[n].LB[d] = parentPtr->LB[d];
			children[n].UB[d] = parentPtr->UB[d];
		}
	}
	double splitVal = 0.5*(parentPtr->LB[dim]+parentPtr->UB[dim]);
	children[0].UB[dim] = splitVal;
	children[1].LB[dim] = splitVal;

	for (n=0; n<2; n++){
		children[n].level = parentPtr->level+1;
		children[n].Npoly = 0;
		children[n].children = NULL;
		children[n].parent = parentPtr;

		children[n].polygons = (double*) MallocOrDie((parentPtr->Npoly)*10*3*sizeof(double));
		children[n].polySizes = (int*) MallocOrDie((parentPtr->Npoly)*sizeof(int));
		children[n].faceInd = (long*) MallocOrDie((parentPtr->Npoly)*sizeof(long));

		children[n].edgeInds = (int*) MallocOrDie(parentPtr->Nedges*2*sizeof(int));
		children[n].edges = (double*) MallocOrDie(parentPtr->Nedges*6*sizeof(double));
		children[n].Nedges = 0;

		children[n].vertInd = (long*) MallocOrDie((parentPtr->Nverts)*sizeof(long));
		children[n].Nverts = 0;
	}

	// faces
	for (n=0; n < parentPtr->Npoly; n++){ // construct polygons for children
		int Nneg = 0, Npos = 0;
		for (k=0; k<parentPtr->polySizes[n]; k++){
			Npos += (parentPtr->polygons[n*10*3+k*3+dim] >= splitVal+eps) ? 1 : 0;
			Nneg += (parentPtr->polygons[n*10*3+k*3+dim] <= splitVal-eps) ? 1 : 0;
		}
		if (Npos == 0){ // second child is empty
			for (k=0; k<parentPtr->polySizes[n]; k++){
				for (d=0; d<3; d++){
					children[0].polygons[(children[0].Npoly)*10*3+k*3+d] \
					= parentPtr->polygons[n*10*3+k*3+d];
				}
			}
			children[0].polySizes[children[0].Npoly] = parentPtr->polySizes[n];
			children[0].faceInd[children[0].Npoly] = parentPtr->faceInd[n];
			children[0].Npoly += 1;
		} else if (Nneg == 0){ // first child is empty
			for (k=0; k<parentPtr->polySizes[n]; k++){
				for (d=0; d<3; d++){
					children[1].polygons[(children[1].Npoly)*10*3+k*3+d] \
					= parentPtr->polygons[n*10*3+k*3+d];
				}
			}
			children[1].polySizes[children[1].Npoly] = parentPtr->polySizes[n];
			children[1].faceInd[children[1].Npoly] = parentPtr->faceInd[n];
			children[1].Npoly += 1;
		} else { // both children are nonempty
			children[0].faceInd[children[0].Npoly] = parentPtr->faceInd[n];
			children[1].faceInd[children[1].Npoly] = parentPtr->faceInd[n];

			children[0].polySizes[children[0].Npoly] = 0;
			children[1].polySizes[children[1].Npoly] = 0;
			for (k=0; k<parentPtr->polySizes[n]; k++){
				k1 = (k+1)%(parentPtr->polySizes[n]);
				// intersection with splitting plane at an internal point of the edge; add new vertex at intersection
				if (parentPtr->polygons[n*10*3+k*3+dim] <= splitVal-eps &&
						parentPtr->polygons[n*10*3+k1*3+dim] >= splitVal+eps){
					for (d=0; d<3; d++){
						children[0].polygons[(children[0].Npoly)*10*3+ \
											 children[0].polySizes[children[0].Npoly]*3+d] \
											 = parentPtr->polygons[n*10*3+k*3+d];
					}
					children[0].polySizes[children[0].Npoly] += 1;

					a = (splitVal-parentPtr->polygons[n*10*3+k1*3+dim])/ \
							(parentPtr->polygons[n*10*3+k*3+dim]-parentPtr->polygons[n*10*3+k1*3+dim]);
					for (d=0; d<3; d++){
						children[0].polygons[(children[0].Npoly)*10*3+ \
											 children[0].polySizes[children[0].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[0].polySizes[children[0].Npoly] += 1;

					for (d=0; d<3; d++){
						children[1].polygons[(children[1].Npoly)*10*3+ \
											 children[1].polySizes[children[1].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[1].polySizes[children[1].Npoly] += 1;
				} else if (parentPtr->polygons[n*10*3+k*3+dim] >= splitVal+eps &&
						parentPtr->polygons[n*10*3+k1*3+dim] <= splitVal-eps){
					for (d=0; d<3; d++){
						children[1].polygons[(children[1].Npoly)*10*3+ \
											 children[1].polySizes[children[1].Npoly]*3+d] \
											 = parentPtr->polygons[n*10*3+k*3+d];
					}
					children[1].polySizes[children[1].Npoly] += 1;
					a = (splitVal-parentPtr->polygons[n*10*3+k1*3+dim])/ \
							(parentPtr->polygons[n*10*3+k*3+dim]-parentPtr->polygons[n*10*3+k1*3+dim]);
					for (d=0; d<3; d++){
						children[1].polygons[(children[1].Npoly)*10*3+ \
											 children[1].polySizes[children[1].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[1].polySizes[children[1].Npoly] += 1;
					for (d=0; d<3; d++){
						children[0].polygons[(children[0].Npoly)*10*3+ \
											 children[0].polySizes[children[0].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[0].polySizes[children[0].Npoly] += 1;
				} else{ // intersection with splitting plane may be only at the edge end point
					if (parentPtr->polygons[n*10*3+k*3+dim] > splitVal-eps) {
						for (d=0; d<3; d++){
							children[1].polygons[(children[1].Npoly)*10*3+ \
												 children[1].polySizes[children[1].Npoly]*3+d] \
												 = parentPtr->polygons[n*10*3+k*3+d];
						}
						children[1].polySizes[children[1].Npoly] += 1;
					}
					if (parentPtr->polygons[n*10*3+k*3+dim] < splitVal+eps) {
						for (d=0; d<3; d++){
							children[0].polygons[(children[0].Npoly)*10*3+ \
												 children[0].polySizes[children[0].Npoly]*3+d] \
												 = parentPtr->polygons[n*10*3+k*3+d];
						}
						children[0].polySizes[children[0].Npoly] += 1;
					}
				}
			}
			children[0].Npoly += 1;
			children[1].Npoly += 1;
		}
	}

	// edges
	for (n=0; n < parentPtr->Nedges; n++){ // construct edges for children
		// reserve for checking consistency of the split of edges
		int Nneg = 0, Npos = 0;
		for (k=0; k<2; k++){
			Npos += (parentPtr->edges[n*6+k*3+dim] >= splitVal+eps) ? 1 : 0;
			Nneg += (parentPtr->edges[n*6+k*3+dim] <= splitVal-eps) ? 1 : 0;
		}
		if (Npos == 0){ // second child gets no edge
			for (k=0; k<2; k++){
				for (d=0; d<3; d++){
					children[0].edges[(children[0].Nedges)*6+k*3+d] \
					= parentPtr->edges[n*6+k*3+d];
				}
			}
			children[0].edgeInds[2*children[0].Nedges] = parentPtr->edgeInds[2*n];
			children[0].edgeInds[2*children[0].Nedges+1] = parentPtr->edgeInds[2*n+1];
			children[0].Nedges += 1;
		} else if (Nneg == 0){ // first child gets no edge
			for (k=0; k<2; k++){
				for (d=0; d<3; d++){
					children[1].edges[(children[1].Nedges)*6+k*3+d] \
					= parentPtr->edges[n*6+k*3+d];
				}
			}
			children[1].edgeInds[2*children[1].Nedges] = parentPtr->edgeInds[2*n];
			children[1].edgeInds[2*children[1].Nedges+1] = parentPtr->edgeInds[2*n+1];
			children[1].Nedges += 1;
		} else { // edge is shared between children

			// intersection with splitting plane at an internal point of the edge; add new vertex at intersection
			if (parentPtr->edges[n*6+dim] <= splitVal-eps &&
					parentPtr->edges[n*6+3+dim] >= splitVal+eps){
				a = (splitVal-parentPtr->edges[n*6+3+dim])/(parentPtr->edges[n*6+dim]-parentPtr->edges[n*6+3+dim]);
				for (d=0; d<3; d++){
					children[0].edges[(children[0].Nedges)*6+d] = parentPtr->edges[n*6+d];
					children[0].edges[(children[0].Nedges)*6+3+d] = a*(parentPtr->edges[n*6+d])\
							+(1-a)*(parentPtr->edges[n*6+3+d]);
					children[1].edges[(children[1].Nedges)*6+d] = children[0].edges[(children[0].Nedges)*6+3+d];
					children[1].edges[(children[1].Nedges)*6+3+d] = parentPtr->edges[n*6+3+d];
				}
				children[0].edgeInds[2*children[0].Nedges] = parentPtr->edgeInds[2*n];
				children[0].edgeInds[2*children[0].Nedges+1] = parentPtr->edgeInds[2*n+1];
				children[1].edgeInds[2*children[1].Nedges] = parentPtr->edgeInds[2*n];
				children[1].edgeInds[2*children[1].Nedges+1] = parentPtr->edgeInds[2*n+1];
				children[0].Nedges += 1;
				children[1].Nedges += 1;

			} else if (parentPtr->edges[n*6+dim] >= splitVal+eps &&
					parentPtr->edges[n*6+3+dim] <= splitVal-eps){
				a = (splitVal-parentPtr->edges[n*6+3+dim])/(parentPtr->edges[n*6+dim]-parentPtr->edges[n*6+3+dim]);
				for (d=0; d<3; d++){
					children[0].edges[(children[0].Nedges)*6+d] = parentPtr->edges[n*6+3+d];
					children[0].edges[(children[0].Nedges)*6+3+d] = a*(parentPtr->edges[n*6+d])\
							+(1-a)*(parentPtr->edges[n*6+3+d]);
					children[1].edges[(children[1].Nedges)*6+d] = children[0].edges[(children[0].Nedges)*6+3+d];
					children[1].edges[(children[1].Nedges)*6+3+d] = parentPtr->edges[n*6+d];
				}
				children[0].edgeInds[2*children[0].Nedges] = parentPtr->edgeInds[2*n];
				children[0].edgeInds[2*children[0].Nedges+1] = parentPtr->edgeInds[2*n+1];
				children[1].edgeInds[2*children[1].Nedges] = parentPtr->edgeInds[2*n];
				children[1].edgeInds[2*children[1].Nedges+1] = parentPtr->edgeInds[2*n+1];
				children[0].Nedges += 1;
				children[1].Nedges += 1;
			} else { // intersection with splitting plane may be only at the edge end point
				if (parentPtr->edges[n*6+dim] > splitVal-eps) {
					for (d=0; d<3; d++){
						children[1].edges[(children[1].Nedges)*6+d] = parentPtr->edges[n*6+d];
						children[1].edges[(children[1].Nedges)*6+3+d] = parentPtr->edges[n*6+3+d];
					}
					children[1].edgeInds[2*children[1].Nedges] = parentPtr->edgeInds[2*n];
					children[1].edgeInds[2*children[1].Nedges+1] = parentPtr->edgeInds[2*n+1];
					children[1].Nedges += 1;
				} else {
					for (d=0; d<3; d++){
						children[0].edges[(children[0].Nedges)*6+d] = parentPtr->edges[n*6+d];
						children[0].edges[(children[0].Nedges)*6+3+d] = parentPtr->edges[n*6+3+d];
					}
					children[0].edgeInds[2*children[0].Nedges] = parentPtr->edgeInds[2*n];
					children[0].edgeInds[2*children[0].Nedges+1] = parentPtr->edgeInds[2*n+1];
					children[0].Nedges += 1;
				}
			}
		}
	}

	// check edge split consistency
	double lenParent, lenChildren[2], v[3];
	lenParent = 0;
	for (n=0; n < parentPtr->Nedges; n++){
		for(d=0; d<3; d++) v[d] = parentPtr->edges[6*n+3+d]-parentPtr->edges[6*n+d];
		lenParent += sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	}

	for (k=0; k<2; k++){
		lenChildren[k] = 0;
		for (n=0; n < children[k].Nedges; n++){
			for(d=0; d<3; d++) v[d] = children[k].edges[6*n+3+d]-children[k].edges[6*n+d];
			lenChildren[k] += sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
		}
	}
	assert(fabs(lenChildren[0]+lenChildren[1]-lenParent) < 1e-6);

	// vertices
	for (n=0; n < parentPtr->Nverts; n++){ // assign parent's vertices to children
		if (points(parentPtr->vertInd[n], dim) > splitVal+eps){
			children[1].vertInd[children[1].Nverts] = parentPtr->vertInd[n];
			children[1].Nverts += 1;
		} else if (points(parentPtr->vertInd[n], dim) < splitVal-eps){
			children[0].vertInd[children[0].Nverts] = parentPtr->vertInd[n];
			children[0].Nverts += 1;
		} else {
			//assert(children[0].Npoly+children[1].Npoly > 0);
			if (children[0].Npoly > 0){
				children[0].vertInd[children[0].Nverts] = parentPtr->vertInd[n];
				children[0].Nverts += 1;
			} else {
				children[1].vertInd[children[1].Nverts] = parentPtr->vertInd[n];
				children[1].Nverts += 1;
			}
		}
	}

	// check consistency of the vertex split
	assert(children[0].Nverts+children[1].Nverts == parentPtr->Nverts);

	int maxInd = 0, maxIndChild[2];
	for (k=0; k<parentPtr->Nverts; k++) maxInd = (maxInd > parentPtr->vertInd[k]) ? maxInd : parentPtr->vertInd[k];
	assert(maxInd < (long) points.n_rows);
	for (n=0; n<2; n++){
		//assert((children[n].Npoly > 0) || (children[n].Nverts == 0));
		maxIndChild[n] = 0;
		for (k=0; k<children[n].Nverts; k++) maxIndChild[n] = (maxIndChild[n] > children[n].vertInd[k]) ? maxInd : children[n].vertInd[k];
		assert(maxIndChild[n] <= maxInd);
	}
	assert((double) fmax(maxIndChild[0], maxIndChild[1]) == (double) maxInd);

	return children;
}


double computeScalarArea(struct Node *nodePtr){ // total area of all intersections with faces
	int n, k, d;
	double a[3], b[3];
	arma::vec areaNormalVector(3);
	double doubleArea = 0;
	for (n=0; n < nodePtr->Npoly; n++){
		for (d=0; d<3; d++) areaNormalVector[d] = 0;
		for (k=2; k < nodePtr->polySizes[n]; k++){
			for (d=0; d<3; d++){
				a[d] = nodePtr->polygons[n*10*3+k*3+d]-nodePtr->polygons[n*10*3+d];
				b[d] = nodePtr->polygons[n*10*3+(k-1)*3+d]-nodePtr->polygons[n*10*3+d];
			}
			areaNormalVector[0] += a[1]*b[2]-b[1]*a[2];
			areaNormalVector[1] += a[2]*b[0]-b[2]*a[0];
			areaNormalVector[2] += a[0]*b[1]-b[0]*a[1];
		}
		doubleArea += sqrt(areaNormalVector[0]*areaNormalVector[0]+
				areaNormalVector[1]*areaNormalVector[1]+
				areaNormalVector[2]*areaNormalVector[2]);
	}
	return doubleArea/2;
}


arma::vec computeAreaNormal(struct Node *nodePtr){ // sum of area vectors of all intersections with faces
	int n, k, d;
	double a[3], b[3];
	arma::vec areaNormalVector(3);
	for (n=0; n<3; n++) areaNormalVector[n] = 0;

	for (n=0; n < nodePtr->Npoly; n++){
		for (k=2; k < nodePtr->polySizes[n]; k++){
			for (d=0; d<3; d++){
				a[d] = nodePtr->polygons[n*10*3+k*3+d]-nodePtr->polygons[n*10*3+d];
				b[d] = nodePtr->polygons[n*10*3+(k-1)*3+d]-nodePtr->polygons[n*10*3+d];
			}
			areaNormalVector[0] += a[1]*b[2]-b[1]*a[2];
			areaNormalVector[1] += a[2]*b[0]-b[2]*a[0];
			areaNormalVector[2] += a[0]*b[1]-b[0]*a[1];
		}
	}
	areaNormalVector[0] /= 2;
	areaNormalVector[1] /= 2;
	areaNormalVector[2] /= 2;

	return areaNormalVector;
}


arma::vec computeQuadform(struct Node *nodePtr){ // sum of quadratic forms for area vectors of all intersections with faces
	int n, k, d;
	double a[3], b[3];
	arma::vec quadformVector(6);
	for (n=0; n<6; n++) quadformVector[n] = 0;
	double doubleArea, c;

	arma::vec areaNormalVector(3);
	for (n=0; n < nodePtr->Npoly; n++){
		for (d=0; d<3; d++) areaNormalVector[d] = 0;
		for (k=2; k < nodePtr->polySizes[n]; k++){
			for (d=0; d<3; d++){
				a[d] = nodePtr->polygons[n*10*3+k*3+d]-nodePtr->polygons[n*10*3+d];
				b[d] = nodePtr->polygons[n*10*3+(k-1)*3+d]-nodePtr->polygons[n*10*3+d];
			}
			areaNormalVector[0] += a[1]*b[2]-b[1]*a[2];
			areaNormalVector[1] += a[2] * b[0] - b[2] * a[0];
			areaNormalVector[2] += a[0] * b[1] - b[0] * a[1];
		}
		doubleArea = sqrt(
			      areaNormalVector[0] * areaNormalVector[0]
				+ areaNormalVector[1] * areaNormalVector[1]
			    + areaNormalVector[2] * areaNormalVector[2]);
		if (doubleArea > 0) {
			c = 0.5 / doubleArea;
			quadformVector[0] += areaNormalVector[0] * areaNormalVector[0] * c;
			quadformVector[1] += areaNormalVector[1] * areaNormalVector[1] * c;
			quadformVector[2] += areaNormalVector[2] * areaNormalVector[2] * c;
			quadformVector[3] += areaNormalVector[0] * areaNormalVector[1] * c;
			quadformVector[4] += areaNormalVector[0] * areaNormalVector[2] * c;
			quadformVector[5] += areaNormalVector[1] * areaNormalVector[2] * c;
		}
	}
	return quadformVector;
}

arma::vec computeEigenvalues(struct Node *nodePtr){ // eigenvalues of the quadform
	arma::vec quadformVector(6);
	quadformVector = computeQuadform(nodePtr);
	arma::mat Q(3,3);
	Q << quadformVector[0] << quadformVector[3] << quadformVector[4] << arma::endr
			<< quadformVector[3] << quadformVector[1] << quadformVector[5] << arma::endr
			<< quadformVector[4] << quadformVector[5] << quadformVector[2] << arma::endr;

	arma::vec eigenvalueVector(3);
	arma::eig_sym(eigenvalueVector, Q);

	return eigenvalueVector;
}


double computeVolumeElement(struct Node *nodePtr){ // sum of area vectors of all intersections with faces
	int n, k, d;
	double a[3], b[3];
	double volumeElement = 0;
	arma::vec areaNormalVector(3);
	for (n=0; n < nodePtr->Npoly; n++){
		for (d=0; d<3; d++) areaNormalVector[d] = 0;
		for (k=2; k < nodePtr->polySizes[n]; k++){
			for (d=0; d<3; d++){
				a[d] = nodePtr->polygons[n*10*3+k*3+d]-nodePtr->polygons[n*10*3+d];
				b[d] = nodePtr->polygons[n*10*3+(k-1)*3+d]-nodePtr->polygons[n*10*3+d];
			}
			areaNormalVector[0] += a[1]*b[2]-b[1]*a[2];
			areaNormalVector[1] += a[2]*b[0]-b[2]*a[0];
			areaNormalVector[2] += a[0]*b[1]-b[0]*a[1];
		}
		for (d=0; d<3; d++) volumeElement += areaNormalVector[d]*(nodePtr->polygons[n*10*3+d]);
	}
	volumeElement /= 6;
	return volumeElement;
}


double computeEdgeDefect(struct Node *nodePtr, edgeMap const &edgeDefects){
	int n, d;
	double defect, totalDefect, len, v[3];
	std::array<int, 2> edge;
	totalDefect = 0;
    for (n=0; n<nodePtr->Nedges; n++){
    	edge[0] = nodePtr->edgeInds[2*n];
    	edge[1] = nodePtr->edgeInds[2*n+1];
    	defect = edgeDefects.find(edge)->second;
    	for (d=0; d<3; d++) v[d] = nodePtr->edges[6*n+3+d]-nodePtr->edges[6*n+d];
    	len = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    	totalDefect += defect*len;
    }
    return totalDefect/2;
}


void fillFeatureDataIter(struct Node *rootPtr,
		int maxlevel,
		long *m,
		int spatialSize,
		SparseGrid &grid,
		std::vector<float> &features,
		std::list<enum FeatureKind> featureList,
		std::vector<double> const &defects,
		edgeMap const &edgeDefects,
		int fillEmpty,
		arma::mat const &points
){
	// given binary tree, iteratively fill sparse grid data and feature data
	if (rootPtr->level == maxlevel) {
		if ((fillEmpty == 1) || (rootPtr->Npoly > 0)
				|| (rootPtr->Nverts > 0) || (rootPtr->Nedges > 0)){
			int d, x[3], n;
			for (d=0; d<3; d++){
				x[d] = floor(0.5*(rootPtr->LB[d]+rootPtr->UB[d]))+spatialSize/2;
				if (x[d] < 0) printf("%d\n", x[d]);
				assert(x[d] >= 0);
				assert(x[d] < spatialSize);
			}

			n = x[0]*spatialSize*spatialSize+x[1]*spatialSize+x[2];
			grid.mp[n] = *m;

			for (const auto& featureKind: featureList) {
				if (featureKind == Bool){
					features.push_back((rootPtr->Npoly > 0) ? 1 : 0);
				} else if (featureKind == ScalarArea){
					double area = computeScalarArea(rootPtr);
					features.push_back((float) area);
				} else if (featureKind == AreaNormal){
					arma::vec areaNormal = computeAreaNormal(rootPtr);
					for (n=0; n<3; n++) features.push_back(areaNormal[n]);
				} else if (featureKind == VolumeElement){
					double volumeElement = computeVolumeElement(rootPtr);
					features.push_back((float) volumeElement);
				} else if (featureKind == QuadForm){
					arma::vec quadform = computeQuadform(rootPtr);
					for (n=0; n<6; n++) features.push_back(quadform[n]);
				} else if (featureKind == EigenValues){
					arma::vec eigenvalues = computeEigenvalues(rootPtr);
					for (n=0; n<3; n++) features.push_back(eigenvalues[n]);
				} else if (featureKind == VertexAngularDefect){
					double totalNodeDefect = 0;
					for (n=0; n < rootPtr->Nverts; n++) {
						assert(rootPtr->vertInd[n] < (int) defects.size());
						totalNodeDefect += defects[rootPtr->vertInd[n]];
					}
					features.push_back(totalNodeDefect);
				} else if (featureKind == EdgeAngularDefect){
					double defect = computeEdgeDefect(rootPtr, edgeDefects);
					features.push_back((float)defect);
				}
			}
			*m += 1;
		}
	} else if ((fillEmpty == 1) || (rootPtr->Npoly > 0) || (rootPtr->Nverts > 0) || (rootPtr->Nedges > 0)){
		splitNode(rootPtr, rootPtr->level%3, points);
		int s;
		for (s=0; s<2; s++){
			fillFeatureDataIter(rootPtr->children+s, maxlevel, m, spatialSize, grid, features,
					featureList, defects, edgeDefects, fillEmpty, points);
			destroy_node(rootPtr->children[s]);
		}
	}
}


void computeAngDefects(arma::mat const &points, std::vector<std::vector<int>> const &surfaces,
		std::vector<double> &defects, int ignoreUnusedVertices){
	unsigned int m;
	defects.clear();
	for (m=0; m<points.n_rows; m++) defects.push_back(0);
	std::vector<int> surface;
	int Nsurf = surfaces.size();
	int n, k, d;
	double v[3][3], norm[3], angle, a;
	for (n=0; n<Nsurf; n++){
		surface = surfaces[n];
		assert((surface[0] != surface[1]) && (surface[0] != surface[2]) && (surface[1] != surface[2]));
		for (k=0; k<3; k++){
			for (d=0; d<3; d++){
				v[k][d] = points(surface[k],d)-points(surface[(k+1)%3],d);
			}
		}
		for (k=0; k<3; k++){
			norm[k] = sqrt(v[k][0]*v[k][0]+v[k][1]*v[k][1]+v[k][2]*v[k][2]);
			assert(norm[k] > 0); // assume no repeated points
		}
		for (k=0; k<3; k++){
			a = -(v[k][0]*v[(k+1)%3][0]+v[k][1]*v[(k+1)%3][1]+v[k][2]*v[(k+1)%3][2])/(norm[k]*norm[(k+1)%3]);
			a = fmax(-1, fmin(1, a));
			angle = acos(a);
			assert(angle == angle); // check for nan
			//assert((angle >= 0) && (angle <= M_PI));
			defects[surface[(k+1)%3]] += angle;
		}
	}
	for (m=0; m<points.n_rows; m++) defects[m] = 2*M_PI-defects[m];

	if (ignoreUnusedVertices == 1){ // if a vertex does not belong to any surface (as happens in ModelNet), set its defect to 0
		std::vector<int> nSurfaces;
		for (n=0; n<(int)points.n_rows; n++) {
			nSurfaces.push_back(0);
		}
		for (n=0; n<(int)surfaces.size(); n++){
			for (d=0; d<3; d++){
				nSurfaces[surfaces[n][d]] += 1;
			}
		}
		for (n=0; n<(int)points.n_rows; n++) {
			if (nSurfaces[n] == 0) defects[n] = 0;
		}
	}
}


int computeAngDefectsEdges(arma::mat const &points,
		std::vector<std::vector<int>> const &surfaces,
		edgeMap &defects){
	defects.clear();
	std::vector<std::array<double, 3>> normals;

	std::array<double, 3> normal;

	int n, d;
	double a[3], b[3], c[3], len, f;

	// surface normals
	for (n=0; n<(int)surfaces.size(); n++){
		for (d=0; d<3; d++) {
			a[d] = points(surfaces[n][1],d)-points(surfaces[n][0],d);
			b[d] = points(surfaces[n][2],d)-points(surfaces[n][0],d);
		}
		for (d=0; d<3; d++) normal[d] = a[(d+1)%3]*b[(d+2)%3]-a[(d+2)%3]*b[(d+1)%3];
		len = sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
		if (len == 0){
			normal[0] = 1;
			normal[1] = 0;
			normal[2] = 0;
			len = 1;
		}
		assert(len > 0);
		for (d=0; d<3; d++) normal[d] /= len;
		normals.push_back(normal);
	}

	// map edges to surfaces
	std::unordered_multimap<std::array<int, 2>, int, PairHash> edge2surfaceMap;
	std::unordered_set<std::array<int, 2>, PairHash> edgeSet;
	std::array<int, 2> edge;
	for (n=0; n<(int)surfaces.size(); n++){
		for (d=0; d<3; d++){
			edge[0] = surfaces[n][d];
			edge[1] = surfaces[n][(d+1)%3];
			edge2surfaceMap.insert(std::pair<std::array<int, 2>, int>(edge, n));
			edgeSet.insert(std::array<int, 2>(edge));
		}
	}

	int exactlyTwoFacesPerEdge = 1;
	// loop over edges and assign angles
	double def;
	for (auto it=edgeSet.begin(); it!=edgeSet.end(); it++){
		edge = *it;
		std::array<int, 2> edgeBack;
		edgeBack[0] = edge[1];
		edgeBack[1] = edge[0];
		if ((edge2surfaceMap.count(edge) != 1) || (edge2surfaceMap.count(edgeBack) != 1)){
		    defects.insert(std::make_pair(edge, 0));
		    exactlyTwoFacesPerEdge = 0;
		} else{
			for (d=0; d<3; d++){
				a[d] = points(edge[1], d)-points(edge[0], d);
				b[d] = normals[edge2surfaceMap.find(edge)->second][d];
				c[d] = normals[edge2surfaceMap.find(edgeBack)->second][d];
			}

			f = (a[0]*b[1]*c[2]+a[1]*b[2]*c[0]+a[2]*b[0]*c[1]
				-a[0]*b[2]*c[1]-a[1]*b[0]*c[2]-a[2]*b[1]*c[0]);

			def = acos(fmin(1, fmax(-1, b[0]*c[0]+b[1]*c[1]+b[2]*c[2])));
			def *= (f >= 0? 1 : -1);
			defects.insert(std::make_pair(edge, def));
		}
	}
	return exactlyTwoFacesPerEdge;
}


void get_features_list(arma::mat const &points,
		std::vector<std::vector<int>> const &surfaces,
		SparseGrid &grid,
		std::vector<float> &features,
		int &nSpatialSites,
		int spatialSize,
		std::list<enum FeatureKind> featureList,
		int splitEmpty){
	/*
	 Input data:
	    points, surfaces: Provided shape data (.off format)
	    spatialSize: Linear size of the cube containing the shape.
	                 The cube is centered at the origin, so each side is [-spatialSize/2, spatialSize/2].
	                 It is assumed that spatialSize is either even or equal to 1.
	                 The cube is divided into (spatialSize x spatialSize x spatialSize) voxels of unit volume (1 x 1 x 1).
	                 It is assumed that the shape has already been appropriately normalized to fit into the cube.
	     featureList: The list of features to be evaluated.
	                 Each voxel is assigned a feature vector of a fixed length equal to the sum of nFeaturesPerVoxel
	                      over all featureKinds in featureList.
	     splitEmpty: 0 or 1.
	                 If 0, only non-empty voxels will be recorded.
	                 If 1, all voxels will be recorded. This case is currently supported only when spatialSize is a power of 2.
	 Output data:
	    grid: Dictionary of nonempty cells.
	          For the voxel with coordinates x,y,z ranging between 0 and spatialSize-1, grid maps x*spatialSize*spatialSize+y*spatialSize+z
	          into the number enumerating that voxel.
	    nSpatialSites: Number of nonempty voxels (equals the size of the dictionary in grid).
	    features: Feature vector of size nSpatialSites*nFeaturesPerVoxel (features for voxel 0, features for voxel 1, ...).

	 */

	assert(points.n_cols == 3);
	assert((splitEmpty == 0) || (splitEmpty == 1));

	grid.mp.clear();
	features.clear();

	unsigned int row;
	unsigned int col;

	// check shape fits in given cube
	double maxAbsVal = 0;
	for (col=0; col<3; col++){
		for (row=0; row<points.n_rows; row++){
			maxAbsVal = fmax(fabs(points(row, col)), maxAbsVal);
		}
	}
	assert((2*(spatialSize/2) == spatialSize) || (spatialSize == 1));
	assert(maxAbsVal <= spatialSize/2.);

	int depth = (int)ceil(log2((double)spatialSize));
	if (spatialSize == 1) assert(depth == 0);
	if (splitEmpty == 1) assert((int) pow(2, depth) == spatialSize);

	// lowest integer bound for spatialSize/2 of the dyadic form 2**n
	int halfSideDyadic = (int) pow(2, depth-1);
	double bound = (double) halfSideDyadic;

	std::vector<double> defects;
	if (std::find(featureList.begin(), featureList.end(), VertexAngularDefect) != featureList.end()){
		computeAngDefects(points, surfaces, defects, 1);
	}

	edgeMap edgeDefects;
	if (std::find(featureList.begin(), featureList.end(), EdgeAngularDefect) != featureList.end()){
		computeAngDefectsEdges(points, surfaces, edgeDefects);
	}

	struct Node rootNode = initRootNode((long) points.n_rows, (long) surfaces.size(), points, surfaces, bound);
	//if (spatialSize > 1) splitIter(&rootNode, 3*depth, points, splitEmpty); // depth in each of 3 dimensions

	long m = 0;
	fillFeatureDataIter(&rootNode, 3*depth, &m, spatialSize, grid, features,
			featureList, defects, edgeDefects, splitEmpty, points);

	nSpatialSites = features.size()/nFeaturesPerVoxel_list(featureList);
	destroy_node(rootNode);
}

extern "C"
void get_features_list_wrap(
		const long Nverts,
		const long Nfaces,
		const double *vertA,
		const long *faceA,
		const int spatialSize,
		int *nSpatialSitesPtr,
		long *sizePtr,
		float **features,
		int **xyz,
		const int splitEmpty,
		const int NfeatureKinds,
		const char **featureL){
	// Wrapper into standard C types, to call get_features_list from Python
	long n, r, r2;
	int k, m;
	arma::mat points0(vertA, 3, Nverts); // column-wise
	arma::mat points = points0.t();
	std::vector<std::vector<int>> surfaces;
	std::vector<int> surface;
	for (k=0; k<Nfaces; k++){
		surface.clear();
		surface.push_back(faceA[3*k]);
		surface.push_back(faceA[3*k+1]);
		surface.push_back(faceA[3*k+2]);
		surfaces.push_back(surface);
	}
	SparseGrid grid;
	std::vector<float> featuresV;
	std::list<enum FeatureKind> featureList = {};
	for (n=0; n<NfeatureKinds; n++) featureList.push_back(str2enum(featureL[n]));
	assert((int)featureList.size() == NfeatureKinds);

	get_features_list(points,
			surfaces,
			grid,
			featuresV,
			*nSpatialSitesPtr,
			spatialSize,
			featureList,
			splitEmpty);

	*sizePtr = featuresV.size();
	*features = (float*) MallocOrDie(featuresV.size()*sizeof(float));
	std::copy(featuresV.begin(), featuresV.end(), *features);
	*xyz = (int*) MallocOrDie(3*(*nSpatialSitesPtr)*sizeof(int));
	n = 0;
	google::dense_hash_map<int64_t, int, std::hash<int64_t>, std::equal_to<int64_t>>::iterator it;
	for (it = grid.mp.begin(); it!=grid.mp.end(); it++) {
		n += 1;
		m = it->second;
		r = it->first;
		(*xyz)[3*m] = (int) r/(spatialSize*spatialSize);
		r2 = r%(spatialSize*spatialSize);
		(*xyz)[3*m+1] = (int) r2/spatialSize;
		(*xyz)[3*m+2] = (int) r2%spatialSize;
		assert(((*xyz)[3*m])*spatialSize*spatialSize+((*xyz)[3*m+1])*spatialSize+((*xyz)[3*m+2]) == r);
	}
	assert(n == *nSpatialSitesPtr);
}


void get_features(arma::mat const &points,
		std::vector<std::vector<int>> const &surfaces,
		SparseGrid &grid,
		std::vector<float> &features,
		int &nSpatialSites,
		int spatialSize,
		enum FeatureKind featureKind,
		int splitEmpty){

	std::list<enum FeatureKind> featureList = {featureKind};
	get_features_list(points,
			surfaces,
			grid,
			features,
			nSpatialSites,
			spatialSize,
			featureList,
			splitEmpty);

}


extern "C"
void freeme(float **features, int **xyz){
	free(*features);
	free(*xyz);
}

int getMesh(char const *path, arma::mat &points, std::vector<std::vector<int>> &surfaces){
	// The locale may cause decimal point error
	setlocale(LC_NUMERIC,"C");
	FILE *meshFile;
	char line[80];
	long n, tmp, Nverts, Nfaces;
	int n0, n1, n2;
	double x, y, z;
	meshFile = fopen(path, "rt");
	if (!meshFile) {
		printf("File not found!\n");
		return 1;
	}
	fgets(line, 80, meshFile);
	if (strlen(line) == 4){
		fgets(line, 80, meshFile);
		sscanf(line, "%ld %ld", &Nverts, &Nfaces);
	}else{
		sscanf(line+3, "%ld %ld", &Nverts, &Nfaces);
	}

	points.set_size(Nverts, 3);
	for (n=0; n<Nverts; n++){
		fgets(line, 80, meshFile);
		sscanf(line, "%lf %lf %lf", &x, &y, &z);
		points(n,0) = x;
		points(n,1) = y;
		points(n,2) = z;
	}
	std::vector<int> curSurf;
	for (n=0; n<Nfaces; n++){
		fgets(line, 80, meshFile);
		sscanf(line, "%ld %d %d %d", &tmp, &n0, &n1, &n2);
		curSurf.clear();
		curSurf.push_back(n0);
		curSurf.push_back(n1);
		curSurf.push_back(n2);
		surfaces.push_back(curSurf);
	}
	fclose(meshFile);
	return 0;
}

void scaleMesh(arma::mat &points, int spatialSize){
	// center and rescale into [-spatialSize,spatialSize]x[-spatialSize,spatialSize]x[-spatialSize,spatialSize]
	double minA[3], maxA[3], shift[3], scale;
	unsigned int n, d;
	for (d=0; d<3; d++){
		minA[d] = points(0,d);
		maxA[d] = points(0,d);
		for (n=0; n<points.n_rows; n++){
			minA[d] = fmin(minA[d], points(n,d));
			maxA[d] = fmax(maxA[d], points(n,d));
		}
		shift[d] = 0.5*(minA[d]+maxA[d]);
	}

	scale = spatialSize/(maxA[0]-minA[0])/1.2;
	for (d=1; d<3; d++){
		scale = fmin(scale, spatialSize/(maxA[d]-minA[d])/1.2);
	}
	for (d=0; d<3; d++){
		for (n=0; n<points.n_rows; n++){
			points(n,d) -= shift[d];
			points(n,d) *= scale;
			assert(points(n,d) > -spatialSize/2.);
			assert(points(n,d) < spatialSize/2.);
		}
	}

}

double getEulerChar(arma::mat const &points, std::vector<std::vector<int>> const &surfaces){
	std::vector<double> defects;
	computeAngDefects(points, surfaces, defects, 1);
	unsigned int n;
	double totalDefect = 0;
	for (n=0; n<points.n_rows; n++) {
		assert(defects[n] == defects[n]); // check for nan
		totalDefect += defects[n];
	}
	return totalDefect/(2*M_PI);
}











