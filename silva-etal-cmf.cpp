/* -----------------------------------------------------------------------
 * silva-etal-cmf.cpp (updated)
 * -----------------------------------------------------------------------
 * This is the main code for detecting copy-move forgery in digital images.
 *
 * The code is part of the copy-move forgery detection method proposed by
 * Ewerton Silva, Tiago Carvalho, Anselmo Ferreira and Anderson Rocha in
 * "Going deeper into copy-move forgery detection: Exloring image telltales
 * via multi-scale analysis and votation processes", Journal of Visual
 * Communication and Image Representation, Vol. 29, pp. 16-32, 2015.
 *
 * UPDATED IN AUGUST, 2017 -----------------------------------------------
 * Updates: - Several changes in array structures, (e.g., we now adopt Mat
 * 				instead of IplImage structure);
 * 			- C interfaces were almost fully switched by the C++ ones;
 * 			- All OpenCV C interfaces were switched by their equivalent in
 * 				C++, such as those to find the minimum bounding rectangle;
 * 			- Some methods were renamed or had their parameters changed.
 * ---------------------------------------------------------------------*/

/* C libraries */
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

/* OpenCV libraries */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* Self libraries */
#include "surf-cmf.h"

/* NAMESPACES */
using namespace cv;
using namespace std;

// ------------------------------------ //
//               STRUCTURES             //
// ------------------------------------ //

// Structure used to create the lexicographical matrix
typedef struct element posmat;
struct element {
	float *desc;	// descriptors
	int cx;			// X position of the center of the patch
	int cy;			// Y position of the center of the patch
	int group;		// group ("0" for source or "1" for destination)
};

// Structure Pairs
typedef struct structpair elpair;
struct structpair {
	int x1, y1;		//(x1, y1) pair
	int x2, y2; 	//(x2, y2) pair
	int subgroup;	// subgroup number (distance-based)
	elpair *next;	// pointer to the next pair
};

// Structure used to create groups of SURF keypoints (orientation-based)
typedef struct keygroup group;
struct keygroup {
	elpair *first;	// pointer to the first pair
	int nsubgroups; // number of subgroups (distance-based)
	int numpairs;	// total number of pairs
};

// Shows the image on the screen
void showImageNew(char winame[], IplImage* img) {
	cvNamedWindow(winame, CV_WINDOW_AUTOSIZE);
	cvShowImage(winame, img);
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow(winame);
}

// ------------------------------------ //
//          STRUCTURE FUNCTIONS         //
// ------------------------------------ //

// Allocates the array of groups
group *allocGroupArray(int nintervals){
	if (360 % nintervals != 0){
		printf("Number of intervals has to be a divisor of 360!\n");
		exit(1);
	}

	group *g = (group*) malloc(nintervals*sizeof(group));	// array of groups

	// Initializations
	for (int i = 0; i < nintervals; i++){
		g[i].first = NULL;
		g[i].nsubgroups = 0;
		g[i].numpairs = 0;
	}
	return g;
}

// Adds a pair to specific group (i-th element)
void addPair(group *g, int i, int dist, Point p1, Point p2){
	// Instantiating the new pair
	elpair *p = (elpair*) malloc(sizeof(elpair));

	// Normal order of the pair
	p->x1 = p1.x; p->y1 = p1.y;
	p->x2 = p2.x; p->y2 = p2.y;

	if (g[i].numpairs == 0){ // the first element inside the group
		p->next = NULL;
		g[i].first = p;
		p->subgroup = g[i].nsubgroups+1;
		g[i].nsubgroups++;
	}
	else {
		elpair *el = g[i].first;
		elpair *pre = g[i].first;
		while (el != NULL){
			// If the pair was already added (in the inverse order)
			if ((el->x1 == p2.x && el->y1 == p2.y) ||
					(el->x2 == p1.x && el->y2 == p1.y)){
				break;
			}
			int flag = 0;
			if (euclidean(el->x1, el->y1, p1.x, p1.y) <= dist &&
					euclidean(el->x2, el->y2, p2.x, p2.y) <= dist){
				flag = 1;
			}
			else if (euclidean(el->x1, el->y1, p2.x, p2.y) <= dist &&
						euclidean(el->x2, el->y2, p1.x, p1.y) <= dist){
				flag = 1;
				// inverting the order of the pair
				p->x1 = p2.x; p->y1 = p2.y;
				p->x2 = p1.x; p->y2 = p1.y;
			}

			if (flag){
				// Adding the pair to an existing subgroup
				if (el == g[i].first){	// first pair
					p->next = el;
					g[i].first = p;
				}
				else {	// subsequent pairs
					pre->next = p;
					p->next = el;
				}
				p->subgroup = el->subgroup;
				break;
			}
			// New subgroup
			else if (el->next == NULL){
				el->next = p;
				p->next = NULL;
				p->subgroup = g[i].nsubgroups+1;	// new subgroup number
				g[i].nsubgroups++;	// updating the total number of subgroups
				break;
			}
			pre = el;		// predecessor pair
			el = el->next;	// successor pair
		} // end-WHILE

	} // end-ELSE

	g[i].numpairs++;	// updating the number of pairs inside the group i
}

// Computes the total number of subgroups
int getTotalNumSubGroups(group *g, int numintervals){
	int totalsg = 0;
	for (int i = 0; i < numintervals; i++){
		totalsg += g[i].nsubgroups;
	}
	return totalsg;
}

// Computes the boundaries for each subgroup and gets the number of correspondences in them
void getSGBoundsNumCorresp(group *g, int *bound, int *bound2, int *npairssg,
		int nintervals, int height, int width){
	int count = 0;

	// Checking groups
	for (int i = 0; i < nintervals; i++){
		elpair *el = g[i].first;

		// Checking subgroups
		for (int j = 1; j <= g[i].nsubgroups; j++){
			// Bounds for group
			int left = width-1, up = height-1;
			int right = 0, bottom = 0;

			// Bounds for the correspondent group
			int left2 = width-1, up2 = height-1;
			int right2 = 0, bottom2 = 0;

			int count2 = 0;		// counter for the pairs in each subgroup

			while (el != NULL){
				// First subgroup
				if(el->x1 < left)
					left = el->x1;	// left boundary
				if(el->y1 < up)
					up = el->y1;	// up boundary
				if(el->x1 > right)
					right = el->x1;	// right boundary
				if(el->y1 > bottom)
					bottom = el->y1;// bottom boundary

				// Second subgroup
				if(el->x2 < left2)
					left2 = el->x2;	// left boundary
				if(el->y2 < up2)
					up2 = el->y2;	// up boundary
				if(el->x2 > right2)
					right2 = el->x2;// right boundary
				if(el->y2 > bottom2)
					bottom2 = el->y2;// bottom boundary

				count2++;		// updating the number of correspondences in the subgroup
				el = el->next;	// next pair
				if (el == NULL || el->subgroup != j)
					break;		// next pair is from another subgroup
			}

			// Filling the arrays of boundaries
			// Subgroup
			bound[count] = left; bound[count+1] = up;
			bound[count+2] = right; bound[count+3] = bottom;

			// Correspondent subgroup
			bound2[count] = left2; bound2[count+1] = up2;
			bound2[count+2] = right2; bound2[count+3] = bottom2;
			count += 4;

			// Filling the array of number of correspondences in each subgroup
			npairssg[(int)(count/4) - 1] = count2;
		}
		free(el);
	}
}

// Updates groups boundaries
void updateGroups(Mat img, int *bound, int *bound2, int ngroups, int growth){
	for (int i = 0; i < 4*ngroups; i += 4){
		// Expanding bounds of the source group
		if (bound[i] - growth >= 0)
			bound[i] -= growth;			// X1
		else bound[i] -= bound[i];

		if (bound[i+1] - growth >= 0)
			bound[i+1] -= growth;		// Y1
		else bound[i+1] -= bound[i+1];

		if (bound[i+2] + growth < img.cols)
			bound[i+2] += growth;				// X2
		else bound[i+2] += img.cols-bound[i+2]-1;

		if (bound[i+3] + growth < img.rows)
			bound[i+3] += growth;				// Y2
		else bound[i+3] += img.rows-bound[i+3]-1;

		// Expanding bounds of the destination group
		if (bound2[i] - growth >= 0)
			bound2[i] -= growth;		// X1
		else bound2[i] -= bound2[i];

		if (bound2[i+1] - growth >= 0)
			bound2[i+1] -= growth;		// Y1
		else bound2[i+1] -= bound2[i+1];

		if (bound2[i+2] + growth < img.cols)
			bound2[i+2] += growth;				// X2
		else bound2[i+2] += img.cols-bound2[i+2]-1;

		if (bound2[i+3] + growth < img.rows)
			bound2[i+3] += growth;				// Y2
		else bound2[i+3] += img.rows-bound2[i+3]-1;
	}
}

// ------------------------------------ //
//         AUXILIARY FUNCTIONS          //
// ------------------------------------ //

// Paints a circle area of the image around a center (cx, cy)
// This function deals with the image boundaries, since
// the circle opencv function does not deal with it
void paintCircle(Mat& map, int bord, int cx, int cy){
	// Creating the circle mask
	Mat mask = Mat::zeros(Size(2*bord+1, 2*bord+1), CV_8UC1);
	circle(mask, Point(bord, bord), bord, Scalar(255), -1, 8, 0);

	// Offsets
	int offsetX = cx-bord;	// offset for the mask (x)
	int offsetY = cy-bord;	// offset for the mask (y)

	// Painting pixels
	uchar elimg;
	for (int i = offsetX; i <= cx+bord; i++){
		for (int j = offsetY; j <= cy+bord; j++){
			elimg = mask.at<uchar>(j-offsetY, i-offsetX);
			if (elimg == 255){
				map.at<uchar>(j, i) = 255;	// setting the pixel values
			}
		} //end-FOR
	} //end-FOR

	// Releasing mask image
	mask.release();
}

// Adds information about an element (patch)
void addPatchLexMat(posmat *lexmat, int pos, float *desc, int cx, int cy, int group){
	lexmat[pos].desc = desc;	// normal histogram
	lexmat[pos].cx = cx;		// x coordinate od the patch
	lexmat[pos].cy = cy;		// y coordinate od the patch
	lexmat[pos].group = group; 	// group
}

// Computes the distance between descriptors
float descriptorDistance(float *desc1, float *desc2, int size){
	float d = 0;
	for (int i = 0; i < size; i++){
		d += (desc1[i] - desc2[i])*(desc1[i] - desc2[i]);
	}
	return sqrt(d);
}

// Enhances boundaries by finding the minimum rectangle that surrounds each group
void enhanceBoundaries(Mat map, Mat& map_enhanced, int *bound, int *bound2,
		int *nDetGroup, int thndet, int nGroups, int factor){

	// Looking for contours in the map image
	for (int i = 0; i < 4*nGroups; i += 4){
		// Checking the number of detections inside the group
		if (nDetGroup[(int)i/4] < thndet)
			continue;

		// Group (source) ----------------------------------------------
		if (bound[i+2]%2 == 0)
			bound[i+2]--;	// bound correction for X
		if (bound[i+3]%2 == 0)
			bound[i+3]--;	// bound correction for Y

		// Delimiting the area of the source group
		int x_s = (int)(bound[i]/factor), y_s = (int)(bound[i+1]/factor);
		int width_s = (int)(bound[i+2]/factor)-x_s, height_s = (int)(bound[i+3]/factor)-y_s;

		// Cropping the image of the source group
		Mat img_s = map(Rect(x_s, y_s, width_s, height_s));

		// Group (destination / correspondence) -------------------------
		if (bound2[i+2]%2 == 0)
			bound2[i+2]--;	// bound2 correction for X
		if (bound2[i+3]%2 == 0)
			bound2[i+3]--;	// bound2 correction for Y

		// Delimiting the area of the destination group
		int x_d = (int)(bound2[i]/factor), y_d = (int)(bound2[i+1]/factor);
		int width_d = (int)(bound2[i+2]/factor)-x_d, height_d = (int)(bound2[i+3]/factor)-y_d;

		// Cropping the image of the destination group
		Mat img_d = map(Rect(x_d, y_d, width_d, height_d));

		// Checking thresholds for the number of white pixels within the groups
		int n_points_s = countNonZero(img_s), n_points_d = countNonZero(img_d);
		if (n_points_s < thndet || n_points_d < thndet)
			continue;

		// Group (source) ----------------------------------------------
		// Getting all white points
		Mat pts_set_s;
		findNonZero(img_s, pts_set_s);

		// Calculating the minimum box around the points
		RotatedRect bound_rect_s = minAreaRect(pts_set_s);
		Point2f rect_pts_s[4]; bound_rect_s.points(rect_pts_s);

		// Points must be in a pointer structure
		Point* bound_pts_s = (Point*) malloc(4*sizeof(Point));
		for (int i = 0; i < 4; i++){
			bound_pts_s[i] = Point(x_s + rect_pts_s[i].x, y_s + rect_pts_s[i].y);
		}

		// Group (destination / correspondence) -------------------------
		// Getting all white points
		Mat pts_set_d;
		findNonZero(img_d, pts_set_d);

		// Calculating the minimum box around the points
		RotatedRect bound_rect_d = minAreaRect(pts_set_d);
		Point2f rect_pts_d[4]; bound_rect_d.points(rect_pts_d);

		// Points must be in a pointer structure to input into fillConvexPoly
		Point* bound_pts_d = (Point*) malloc(4*sizeof(Point));
		for (int i = 0; i < 4; i++){
			bound_pts_d[i] = Point(x_d + rect_pts_d[i].x, y_d + rect_pts_d[i].y);
		}

		// Drawing and filling the minimum rectangles (boxes) on the image
	    fillConvexPoly(map_enhanced, bound_pts_s, 4, Scalar(255), 8);	// source
	    fillConvexPoly(map_enhanced, bound_pts_d, 4, Scalar(255), 8);	// destination

		// Memory deallocation
	    free(bound_pts_s);
	    free(bound_pts_d);
		img_s.release();
		img_d.release();
	}
}

// ------------------------------------ //
//           SORTING FUNCTIONS          //
// ------------------------------------ //

// QuickSort (used along with RadixSort)
void quickSort(posmat *lexmat, int pos, int left, int right) {
      int i = left, j = right;
      float pivot = lexmat[(int)((left + right)/2)].desc[pos];
      posmat el;

      // Partition
      while (i <= j) {
    	  while (lexmat[i].desc[pos] < pivot)
    		  i++;
    	  while (lexmat[j].desc[pos] > pivot)
    		  j--;
    	  if (i <= j) {
    		  el = lexmat[i];
    		  lexmat[i] = lexmat[j];
    		  lexmat[j] = el;
    		  i++; j--;
		}
      }

      // Recursion
      if (left < j)
            quickSort(lexmat, pos, left, j);
      if (i < right)
            quickSort(lexmat, pos, i, right);
}

// RadixSort (used along with QuickSort to sort the lexicographic matrix)
void radixSort(posmat *lexmat, int size, int ncol){
	quickSort(lexmat, 0, 0, size-1);
	for(int i = 1; i < ncol; i++){
		for (int j = 0; j < size-1; j++){
			int begin = j, end = 0;
			while (j < size-1 && (lexmat[j].desc[i-1] == lexmat[j+1].desc[i-1])){
				end++;
				j++;
			}
			quickSort(lexmat, i, begin, begin+end);
		}
	}
}

// ------------------------------------ //
//         DESCRIPTOR FUNCTIONS         //
// ------------------------------------ //

// Calculates the normalized patch's description (whole concentric circles)
// For 1-channel (i.e, Gray) images, only!
float* getDescriptionGray(Mat img, int radius, int cx, int cy){
	int factor = radius/4;

	// Creating the circle mask 0 (minimum radius)
	Mat mask0 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask0, Point(radius, radius), radius-3*factor, Scalar(255), -1, 8, 0);

	// Creating the circle mask 1
	Mat mask1 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask1, Point(radius, radius), radius-2*factor, Scalar(255), -1, 8, 0);

	// Creating the circle mask 2
	Mat mask2 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask2, Point(radius, radius), radius-factor, Scalar(255), -1, 8, 0);

	// Creating the circle mask 3 (maximum radius)
	Mat mask3 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask3, Point(radius, radius), radius, Scalar(255), -1, 8, 0);

	// Memory allocation
	float *desc = (float*) calloc(4, sizeof(float));

	// Iterating over the patch
	int offsetX = cx-radius;	// offset for x
	int offsetY = cy-radius;	// offset for y
	int total0 = 0, total1 = 0, total2 = 0, total3 = 0;
	uchar elmask, elimg;

	for (int i = cx-radius; i <= cx+radius; i++){
		for (int j = cy-radius; j <= cy+radius; j++){
			// Computing values for mask 3
			elmask = mask3.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position

			elimg = img.at<uchar>(j, i);	// getting the pixel values (grayscale)

			desc[3] += elimg;
			total3++;

			// Computing values for mask 2
			elmask = mask2.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position
			desc[2] += elimg;
			total2++;

			// Computing values for mask 1
			elmask = mask1.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position
			desc[1] += elimg;
			total1++;

			// Computing values for mask 0
			elimg = mask0.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position
			desc[0] += elimg;
			total0++;
		}
	}

	// Calculating the mean of each circle
	desc[0] /= (total0);
	desc[1] /= (total1);
	desc[2] /= (total2);
	desc[3] /= (total3);

	// Releasing masks
	mask0.release();
	mask1.release();
	mask2.release();
	mask3.release();

	return desc;
}

// Calculates the normalized patch's description (whole concentric circles)
// The mean of each channel is taken individually
// For 3-channel (i.e, RGB, HSV etc.) images, only!
float* getDescriptionColor(Mat img, int radius, int cx, int cy){
	int factor = radius/4;

	// Creating the circle mask 0 (minimum radius)
	Mat mask0 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask0, Point(radius, radius), radius-3*factor, Scalar(255), -1, 8, 0);

	// Creating the circle mask 1
	Mat mask1 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask1, Point(radius, radius), radius-2*factor, Scalar(255), -1, 8, 0);

	// Creating the circle mask 2
	Mat mask2 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask2, Point(radius, radius), radius-factor, Scalar(255), -1, 8, 0);

	// Creating the circle mask 3 (maximum radius)
	Mat mask3 = Mat::zeros(Size(2*radius+1, 2*radius+1), CV_8UC1);
	circle(mask3, Point(radius, radius), radius, Scalar(255), -1, 8, 0);

	// Memory allocation
	float *desc = (float*) calloc(12, sizeof(float));

	// Iterating over the patch
	int offsetX = cx-radius;	// offset for x
	int offsetY = cy-radius;	// offset for y
	int total0 = 0, total1 = 0, total2 = 0, total3 = 0;
	uchar elmask; Vec3b elimg;

	// Getting the max and min values
	float max = 0.0, min = 255.0;

	for (int i = offsetX; i <= cx+radius; i++){
		for (int j = offsetY; j <= cy+radius; j++){

			if (mask3.at<uchar>(j-offsetY, i-offsetX) == 255){
				elimg = img.at<Vec3b>(j, i);	// getting the pixel values
				// MAX
				if (max < elimg.val[0])
					max = elimg.val[0];
				if (max < elimg.val[1])
					max = elimg.val[1];
				if (max < elimg.val[2])
					max = elimg.val[2];
				// MIN
				if (min > elimg.val[0])
					min = elimg.val[0];
				if (min > elimg.val[1])
					min = elimg.val[1];
				if (min > elimg.val[2])
					min = elimg.val[2];
			}
		}
	}

	for (int i = offsetX; i <= cx+radius; i++){
		for (int j = offsetY; j <= cy+radius; j++){
			// Computing values for mask 3
			elmask = mask3.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position

			elimg = img.at<Vec3b>(j, i);	// getting the pixel values

			desc[3] += (elimg.val[0]-min)/(max-min);	// Blue / Hue
			desc[7] += (elimg.val[1]-min)/(max-min);	// Green / Saturation
			desc[11] += (elimg.val[2]-min)/(max-min);	// Red / Value
			total3++;

			// Computing values for mask 2
			elmask = mask2.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position
			desc[2] += (elimg.val[0]-min)/(max-min);	// Blue	/ Hue
			desc[6] += (elimg.val[1]-min)/(max-min);	// Green / Saturation
			desc[10] += (elimg.val[2]-min)/(max-min);	// Red / Value
			total2++;

			// Computing values for mask 1
			elmask = mask1.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position
			desc[1] += (elimg.val[0]-min)/(max-min);	// Blue / Hue
			desc[5] += (elimg.val[1]-min)/(max-min);	// Green / Saturation
			desc[9] += (elimg.val[2]-min)/(max-min);	// Red / Value
			total1++;

			// Computing values for mask 0
			elmask = mask0.at<uchar>(j-offsetY, i-offsetX);
			if (elmask != 255)
				continue;	// if the mask value is 0, do not consider this position
			desc[0] += (elimg.val[0]-min)/(max-min);	// Blue / Hue
			desc[4] += (elimg.val[1]-min)/(max-min);	// Green / Saturation
			desc[8] += (elimg.val[2]-min)/(max-min);	// Red / Value
			total0++;
		}
	}

	// Calculating the mean of each circle and each channel
	//		Blue	----		Green	    ----	 Red
	desc[0] /= (total0); desc[4] /= (total0); desc[8] /= (total0);	// mask 3
	desc[1] /= (total1); desc[5] /= (total1); desc[9] /= (total1);	// mask 2
	desc[2] /= (total2); desc[6] /= (total2); desc[10] /= (total2);	// mask 1
	desc[3] /= (total3); desc[7] /= (total3); desc[11] /= (total3);	// mask 0

	// Releasing masks
	mask0.release();
	mask1.release();
	mask2.release();
	mask3.release();

	return desc;
}

// ------------------------------------ //
//        CMF DETECTION FUNCTION        //
// ------------------------------------ //

// Scans the final Lexicographical Matrix towards similar patches
void detectLexicog(Mat img, Mat& map, posmat *lexmat, int *ndet, int bord,
		int size, int descsize, int neigh, float tsim, float mindist){

	for (int i = 0; i < size-1; i++){
		float dist, magnitude;
		int j = i+1;
		while (j < size && j <= i+neigh){
			// Checking whether the blocks are from the same region (group)
			if (lexmat[i].group == lexmat[j].group){
				j++;
				continue;
			}

			// Calculating the magnitude (physical distance) between patches
			magnitude = euclidean(lexmat[i].cx, lexmat[i].cy, lexmat[j].cx, lexmat[j].cy);
			if (magnitude <= mindist){
				j++;
				continue;
			}

			// Computing the distance between histograms
			dist = descriptorDistance(lexmat[i].desc, lexmat[j].desc, descsize);

			// Checking the threshold
			if (dist <= tsim){
				uchar el = map.at<uchar>(lexmat[i].cy, lexmat[i].cx);
				if (el != 255){
					(*ndet)++;	// updating the number of detections in the group
					paintCircle(map, bord, lexmat[i].cx, lexmat[i].cy);
					paintCircle(map, bord, lexmat[j].cx, lexmat[j].cy);
				}
			}
			j++;
		}
	}
}

// Performs the pyramidal decomposition and stores the images in the specified array
void pyrDecompose(Mat img, vector<Mat>& arrayimg, int numlevels){
	arrayimg.push_back(img);
	for (int i = 1; i < numlevels; i++){
		Mat dec_img = Mat::zeros(Size(arrayimg[i-1].cols/2, arrayimg[i-1].rows/2), img.type());
		pyrDown(arrayimg[i-1], dec_img, Size(arrayimg[i-1].cols/2, arrayimg[i-1].rows/2));
		arrayimg.push_back(dec_img.clone());
	}
}

// Performs the votation scheme to find the final image map
void votationMap(vector<Mat> arraymap, Mat& map_final, int numlevels, int thpyr){
	// Iterating over all grayscale maps in the pyramid
	for (int i = 0; i < arraymap[0].cols; i++){
		for (int j = 0; j < arraymap[0].rows; j++){

			if (arraymap[0].at<uchar>(j, i) != 0){
				int count = 1;
				for (int n = 1; n < numlevels; n++){
					if (arraymap[n].at<uchar>(j, i) != 0){
						count++;
					}
				} // end-FOR

				if (count >= thpyr){
					map_final.at<uchar>(j, i) = 255;
				}
			} // end-IF

		} // end-FOR
	} // end-FOR

}

// ------------------------------------ //
//				MAIN FUNCTION			//
// 		DETECTS COPY-MOVE FORGERIES		//
// ------------------------------------ //

Mat detectCMF(char imgpath[]){
	// Reading the image -------------------------------------------------------------------
	Mat img_orig = imread(imgpath);
	int n_channels = img_orig.channels(); // number of color channels

	// Converting to HSV -------------------------------------------------------------------
	Mat img = Mat::zeros(Size(img_orig.cols, img_orig.rows), img_orig.type());
	cvtColor(img_orig, img, CV_BGR2HSV);	// RGB, BGR, GRAY, HSV, YCrCb, XYZ, Lab, Luv, HLS
	img_orig.release();

	// Converting to Grayscale -------------------------------------------------------------
	Mat img_gray = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);
	cvtColor(img, img_gray, CV_BGR2GRAY);	// to extract SURF keypoints

	// Important variables for SURF part ---------------------------------------------------
	int thessian = 0;		// hessian threshold (0 for the max. number of keypoints)
	float thfactor = 0.6;	// factor threshold to compare two keypoints
	int tphy = 50;			// minimum physical distance between keypoints
	int tphy2 = 50;			// maximum distance between a keypoint and its closest
							// matching pair in a subgroup
	int nintervals = 20;	// number of orientation intervals
	int growth = 10;		// subgroup size maximum growth in each direction

	// Important variables for Detection part ----------------------------------------------
	int psize = 9; 			// patch size
	int bord = psize/2;		// border size
	int neigh = 5;			// number of neighboring lines to check
	float tsim = 0.05;		// threshold distance to evaluate similarity
	float mindist = 30;		// minimum physical distance between similar patches
	int thndet = 3;			// minimum number of detections in the group
	int thnpairs = 3;		// NEW: minimum number of pairs of keypoints in a subgroup
	int numlevels = 3;		// number of levels in the pyramidal decomposition
	int thpyr = 2;			// minimum number of detection through the scales

	// ------------------------------------------------------------------ //
	// PART I - SURF correspondences detection and groups of keypoints	  //
	// ------------------------------------------------------------------ //

	// Extracting SURF keypoints and correspondences
	vector<Point> pairs;
	getSURFKeypointsPairs(img_gray, pairs, thessian, thfactor, tphy);

	// Creating groups of keypoint correspondences
	group *g = allocGroupArray(nintervals);	// groups of keypoints

	for (int i = 0; i < (int)pairs.size(); i+=2){
		// Computing the angle between the correspondent keypoints
		float dx = abs(pairs[i+1].x - pairs[i].x);
		float dy = abs(pairs[i+1].y - pairs[i].y);
		float angle = (atan(dy/dx)*180.0)/M_PI;
		if (angle < 0){	// it seems that this is unnecessary
			angle += 360.0;	// atan can return negative angles
		}

		// Adding the keypoint to a group
		int ngroup = (int)(angle*nintervals)/360;
		addPair(g, ngroup, tphy2, pairs[i], pairs[i+1]);
	}

	// Getting the total number of subgroups
	int totalsg = getTotalNumSubGroups(g, nintervals);

	// Getting subgroups boundaries and the number of correspondences in each subgroup
	int *bound = (int*) malloc((4*totalsg)*sizeof(int)); 	// bounds for source subgroups
	int *bound2 = (int*) malloc((4*totalsg)*sizeof(int));	// bounds for destination subgroups
	int *npairssg = (int*) malloc(totalsg*sizeof(int));		// number of correspondences in each subgroup
	getSGBoundsNumCorresp(g, bound, bound2, npairssg, nintervals, img_gray.rows, img_gray.cols);

	// Updating keypoints groups (expanding each according to the growth rate)
	updateGroups(img, bound, bound2, totalsg, growth);

	// Freeing memory
	free(g);
	img_gray.release();

	// ------------------------------------------------------------------- //
	// PART II - Traditional CMF Detection method over the groups found    //
	// ------------------------------------------------------------------- //
	// Using pyramid decomposition and getting an array of images/map
	vector<Mat> arrayimg, arraymap;	// structures for image and map pyramids
	pyrDecompose(img, arrayimg, numlevels);	// pyramidal decomposition

	img.release();	// releasing the color image

	// Running the method for each image level
	for (int n = 0; n < numlevels; n++){
		Mat map = Mat::zeros(Size(arrayimg[n].cols, arrayimg[n].rows), CV_8UC1);	// Map image
		int *nDetGroup = (int*) calloc(totalsg, sizeof(int));	// number of detections per group
		int factor = (int) powf(2.0, (float)n);	// factor to resize the group

		// Changing the distance by a factor
		mindist /= factor;

		// Checking forgeries in each group
		for (int k = 0; k < 4*totalsg; k += 4){
			// Checking if the subgroups have a minimum number of correspondence pairs
			if (npairssg[(int)(k/4)] < thnpairs)
				continue;

			// Finding the number of lines of the lexicographical matrix (group 1)
			int g1width = (int)(bound[k+2]-bound[k]+1)/factor;
			int g1height = (int)(bound[k+3]-bound[k+1]+1)/factor;
			int size1 = (g1height-psize+1)*(g1width-psize+1);

			// Finding the number of lines of the lexicographical matrix (group 2)
			int g2width = (int)(bound2[k+2]-bound2[k]+1)/factor;
			int g2height = (int)(bound2[k+3]-bound2[k+1]+1)/factor;
			int size2 = (g2height-psize+1)*(g2width-psize+1);

			// Checking whether the group has at least the size of the sliding window
			if (g1height < psize || g1width < psize ||
					g2height < psize || g2width < psize){
				continue;
			}

			// Allocating the final lexicographical matrix
			int size = size1 + size2;
			posmat *lexmat = (posmat*) malloc(size*sizeof(posmat));

			// Iterating over the image
			int next = 0;	// stores the next position of lexmat to be filled
			int descsize = 0;	// descriptor size

			// SOURCE GROUP ------------------------------------------------------------------------
			int boundX = (int)bound[k]/factor;
			int boundY = (int)bound[k+1]/factor;

			for (int i = bord+boundX; i < boundX+g1width-bord; i++) {
				for (int j = bord+boundY; j < boundY+g1height-bord; j++) {
					float *desc;
					if (n_channels > 1){
						descsize = 12;
						desc = getDescriptionColor(arrayimg[n], bord, i, j);
					}
					else {
						descsize = 4;
						desc = getDescriptionGray(arrayimg[n], bord, i, j);
					}
					addPatchLexMat(lexmat, next, desc, i, j, 0);
					next++;
				} // end-FOR
			} // end-FOR

			// DESTINATION/CORRESPONDENCE GROUP -----------------------------------------------------
			int boundX2 = (int)bound2[k]/factor;
			int boundY2 = (int)bound2[k+1]/factor;

			for (int i = bord+boundX2; i < boundX2+g2width-bord; i++) {
				for (int j = bord+boundY2; j < boundY2+g2height-bord; j++) {
					float *desc;
					if (n_channels > 1){
						descsize = 12;
						desc = getDescriptionColor(arrayimg[n], bord, i, j);
					}
					else {
						descsize = 4;
						desc = getDescriptionGray(arrayimg[n], bord, i, j);
					}
					addPatchLexMat(lexmat, next, desc, i, j, 1);
					next++;
				} // end-FOR
			} // end-FOR

			// Sorting the lexicographical matrix
			radixSort(lexmat, size, descsize);

			// Looking for similar patches in the lexicographical matrix
			int nDet = 0;
			detectLexicog(arrayimg[n], map, lexmat, &nDet, bord, size, descsize, neigh,
					tsim, mindist);

			// Updating the number of detections in the current group
			nDetGroup[(int)(k/4)] += nDet;

			// Lexicographical matrix structure deallocation
			for (int i = 0; i < size; i++){
				free(lexmat[i].desc);
			}
			free(lexmat);

		}

		// Drawing the new enhanced boundaries
		Mat map_enhanced = Mat::zeros(Size(arrayimg[n].cols, arrayimg[n].rows), CV_8UC1);
		enhanceBoundaries(map, map_enhanced, bound, bound2, nDetGroup, thndet, totalsg, factor);

		// Inserting a copy of the enhanced map in the arraymap
		if (n > 0){ // this is not done on the original image
			// Resizing the cloned map and putting it back into the array
			resize(map_enhanced, map_enhanced, Size(arraymap[0].cols, arraymap[0].rows), 0, 0, INTER_LINEAR);
		}
		arraymap.push_back(map_enhanced.clone());

		// Releasing images and freeing other structures
		map.release();
		map_enhanced.release();
		arrayimg[n].release();
		free(nDetGroup);

	} // end-FOR

	// ------------------------------------------------------------------- //
	// PART III - Votation and Closing steps 							   //
	// ------------------------------------------------------------------- //

	// Calculating the final map by a votation scheme
	Mat map_final = Mat::zeros(Size(arraymap[0].cols, arraymap[0].rows), CV_8UC1);
	votationMap(arraymap, map_final, numlevels, thpyr);

	// Deallocating memory
	free(bound);
	free(bound2);
	free(npairssg);

	// Releasing map images
	for (int n = 0; n < numlevels; n++){
		arraymap[n].release();
	}

	return map_final;
}


// ------------------------------------ //
//             MAIN FUNCTION			//
// ------------------------------------ //

int main(int argc, char *argv[]){
	Mat map_final = detectCMF(argv[1]);
	imshow("Detection result", map_final);
	waitKey(0);
	return 0;
}
