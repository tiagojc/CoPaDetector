/* -----------------------------------------------------------------------
 * silva-etal-cmf.h	(updated)
 * -----------------------------------------------------------------------
 * This library contains the main method's signature to be called for
 * detecting copy-move forgeries in digital images.
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
 *  Defining the functions
 * ----------------------------- */
extern IplImage* detectCMF(char imgpath[]);
