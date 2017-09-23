/* -----------------------------------------------------------------------
 * surf-cmf.h	(updated)
 * -----------------------------------------------------------------------
 * This library contains methods to extract SURF keypoints from
 * an image, and to get only the surf correspondences.
 *
 * The code is part of the copy-move forgery detection method proposed by
 * Ewerton Silva, Tiago Carvalho, Anselmo Ferreira and Anderson Rocha in
 * "Going deeper into copy-move forgery detection: Exloring image telltales
 * via multi-scale analysis and votation processes", Journal of Visual
 * Communication and Image Representation, Vol. 29, pp. 16-32, 2015.
 *
 * UPDATED IN AUGUST, 2017 -----------------------------------------------
 * Updates: - Renamed method
 * ---------------------------------------------------------------------*/

/* -----------------------------
 *  Defining the SURF functions
 * ----------------------------- */

using namespace std;
using namespace cv;

extern void getSURFKeypointsPairs(Mat image, vector<Point>& pairs, int thhessian,
		float thfactor, float thdist);
extern float euclidean(int x1, int y1, int x2, int y2);
