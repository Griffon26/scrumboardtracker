// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage (it's got it all folks) to find\n"
    "squares in a list of images pic1-6.png\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int N = 2;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());

        Mat hsv(image.size(), image.depth());
        cvtColor( image, hsv, CV_BGR2HSV );

        vector<Mat> hsv_planes;
        Mat sat;
        split(hsv, hsv_planes);
        sat = hsv_planes[1];

        {
          imshow(wndname, sat);
          int c;
          while((c = waitKey()) > 255);
        }

        inRange(sat, 0, 25, gray0);

        {
          imshow(wndname, gray0);
          int c;
          while((c = waitKey()) > 255);
        }

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            cout << "  Threshold level " << l << endl;


            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, 50, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            // find contours and store them all as a list
            findContours(gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

            if(contours.size() != 0)
            {
              Mat contourImage(image.size(), CV_8UC3, Scalar(0,0,0));
              contourImage = Scalar(0,0,0);
              cout << "Drawing " << contours.size() << " contours" << endl;
              drawContours(contourImage, contours, -1, Scalar(255,0,255), 1, 8, hierarchy, 2);
              imshow(wndname, contourImage);

              int c;
              while((c = waitKey()) > 255);
            }

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                {
                  /* FIXME: This is an experiment to see if the contourtree can still be used.
                   * If so it should be easier to match rectangles.
                   */
                  CvMemStorage* storage = cvCreateMemStorage(1000);

                  CvSeq* myKeypointSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(KeyPoint),storage);
                  // Create the seq at the location storage

                  for (size_t ii=0; ii < contours.size(); ii++) {
                      int* added = (int*)cvSeqPush(myKeypointSeq,&(contours[i][ii]));
                      // Should add the KeyPoint in the Seq
                  }

                  CvMemStorage* treeStorage = cvCreateMemStorage(1000);
                  cvCreateContourTree(myKeypointSeq, treeStorage, 0);
                }

                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {



                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.2 )
                        squares.push_back(approx);
                }
            }
        }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);
        cout << "square " << i << endl;
        for( int j = 0; j < 4; j++)
        {
          int otherj = (j + 1) % 4;
          Point diff = (squares[i][j] - squares[i][otherj]);
          double distance = sqrt(diff.x*diff.x + diff.y*diff.y);
          cout << " edge " << j << " distance is " << distance << endl;
        }
    }


    imshow(wndname, image);
}


int main(int /*argc*/, char** /*argv*/)
{
    static const char* names[] = { "scrumboardphoto.jpg", 0 };
    help();
    namedWindow( wndname, 1 );
    moveWindow(wndname, 100, 100);
    vector<vector<Point> > squares;

    for( int i = 0; names[i] != 0; i++ )
    {
        Mat image = imread(names[i], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findSquares(image, squares);
        drawSquares(image, squares);

        int c;
        while((c = waitKey()) > 255);
        cout << "Key was" << c << endl;
        if( (char)c == 27 )
            break;
    }

    return 0;
}
