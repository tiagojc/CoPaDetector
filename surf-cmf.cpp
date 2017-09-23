/* -----------------------------------------------------------------------
 * surf-cmf.cpp	(updated)
 * -----------------------------------------------------------------------
 * This is the code for extracting SURF keypoints and decriptors and
 * finding correspondences between them.
 *
 * The code is part of the copy-move forgery detection method proposed by
 * Ewerton Silva, Tiago Carvalho, Anselmo Ferreira and Anderson Rocha in
 * "Going deeper into copy-move forgery detection: Exloring image telltales
 * via multi-scale analysis and votation processes", Journal of Visual
 * Communication and Image Representation, Vol. 29, pp. 16-32, 2015.
 *
 * UPDATED IN AUGUST, 2017 -----------------------------------------------
 * Updates: - Several changes in array structures (e.g., vector<KeyPoints>
 * 				instead of the previously used pairs** array);
 * 			- Keypoints matching is now done by means of OpenCV routines
 * 				like KNN and BFMatcher directly;
 * 			- All OpenCV C interfaces were switched by their equivalent in
 * 				C++, such as in the SURF keypoints extraction part;
 * 			- Some methods were added/renamed/removed.
 * ---------------------------------------------------------------------*/

/* OpenCV libraries */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

/* C libraries */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>

/* Self libraries */
#include "surf-cmf.h"

/* NAMESPACES */
using namespace std;
using namespace cv;

// Calculates the Euclidean Distance between two points (x1, y1) and (x2, y2)
float euclidean(int x1, int y1, int x2, int y2){
	return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

// Copies the specified row from a Mat structure to a vector
void copyRow(Mat mat, int index, vector<float>& row){
	for (int j = 0; j < mat.cols; j++){
		row.push_back(mat.at<float>(index, j));
	}
}

// Compares two SURF descriptors
double compareSURFDescriptors(vector<float> desc1, vector<float> desc2,
		double best, int length) {

	double total_cost = 0;
	assert(length % 4 == 0);	// 64-d, 128-d (used) or 256-d

	// Iterating over the descriptor vectors
	for (int i = 0; i < length; i += 4) {
		double t0 = desc1[i] - desc2[i];
		double t1 = desc1[i + 1] - desc2[i + 1];
		double t2 = desc1[i + 2] - desc2[i + 2];
		double t3 = desc1[i + 3] - desc2[i + 3];
		total_cost += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
		if (total_cost > best)
			break;	// stop if it's higher than best, because
					// these descriptors are not similar
	}

	return total_cost;
}

// Uses a Naive Nearest Neighbor search with Ratio test to find
// correspondent pairs between keypoint descriptors
void findPairsByNaiveKNN(Mat descriptors, vector<KeyPoint> keypoints, vector<Point>& pairs,
		float thfactor, float thdist){
	vector< vector<DMatch> > matches;
	BFMatcher matcher;
	matcher.knnMatch(descriptors, descriptors, matches, 3);  // Find three nearest matches
															// the first one is to the current keypoint itself
	// Checking the quality of the matches
	// by using the Ratio Test (thfactor value)
	for (int i = 0; i < (int)matches.size(); i++){
		if (matches[i].size() == 3){
			if (matches[i][1].distance < thfactor * matches[i][2].distance){
				Point orig = keypoints[matches[i][0].queryIdx].pt;
				Point dest = keypoints[matches[i][1].trainIdx].pt;

				float dist = euclidean(cvRound(orig.x), cvRound(orig.y),
						cvRound(dest.x), cvRound(dest.y));
				if (dist > thdist){
					pairs.push_back(Point(cvRound(orig.x),cvRound(orig.y)));
					pairs.push_back(Point(cvRound(dest.x),cvRound(dest.y)));
				}
			}
	    }
	}
}

// Extracts and gets all SURF keypoints pairs
void getSURFKeypointsPairs(Mat image, vector<Point>& pairs, int thhessian,
		float thfactor, float thdist) {
	// Variables
	vector<KeyPoint> keypoints;
	Mat descriptors;

	// Performing the SURF keypoints extraction
	Ptr<Feature2D> surf = xfeatures2d::SURF::create(thhessian,
			4, 3, true, false); // true means extended descriptor (128-d)
	surf->detectAndCompute(image, Mat(), keypoints, descriptors);

	// Finding correspondences
	findPairsByNaiveKNN(descriptors, keypoints, pairs, thfactor, thdist);
}
