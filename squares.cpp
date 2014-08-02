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
#include <unistd.h>

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

static double eucldistance(Point p1, Point p2)
{
    return norm(p1 - p2);
}

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

static double angle2( Point pt1, Point pt0, Point pt2)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

class MyFilter: public BaseFilter
{
public:
    MyFilter()
    {
      this->ksize = Size(4,4);
      this->anchor = Point(0,0);
    }

    MyFilter(Mat templ)
    {
      this->ksize = templ.size();
      this->anchor = Point(0,0);
    }

    // To be overriden by the user.
    //
    // runs a filtering operation on the set of rows,
    // "dstcount + ksize.height - 1" rows on input,
    // "dstcount" rows on output,
    // each input row has "(width + ksize.width-1)*cn" elements
    // each output row has "width*cn" elements.
    // the filtered rows are written into "dst" buffer.
    virtual void operator()(const uchar** src, uchar* dst, int dststep,
                            int dstcount, int width, int cn)
    {
      static uchar line = 0;

      assert(cn == 3);
      cout << "dststep, dstcount, width, cn: " << dststep << " " << dstcount << " " << width << " " << cn << endl;

      for(int y = 0; y < dstcount; y++)
      {
        for(int x = 0; x < width; x++)
        {
          const uchar *s = (src[y] + x * cn);
          uchar *d = (dst + (dststep * y) + x * cn);

          uchar color = (x < 10) ? line : y * 20;

          d[0] = (s[0] + s[3]) / 2;
          d[1] = (s[1] + s[4]) / 2;
          d[2] = (s[2] + s[5]) / 2;
        }
      }

      line += 5;

    }
};



static void findSquares( const Mat& image )
{
    cout << "image type: " << getImageType(image.type()) << endl;



#if 0
    Mat output(image.size(), image.type());
    Ptr<BaseFilter> pfilter = new MyFilter();
    FilterEngine engine(pfilter, 0, 0, image.type(), image.type(), image.type());
    engine.apply(image, output);
    {
        imshow(wndname, output);
        int c;
        while((c = waitKey()) > 255);
    }

#endif




    Mat gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    Mat downscaled, denoised;
    pyrDown(image, downscaled, Size(image.cols/2, image.rows/2));
    pyrUp(downscaled, denoised, image.size());
    cout << "denoised type: " << getImageType(denoised.type()) << endl;

    denoised = image;
    {
        imshow(wndname, denoised);
        int c;
        while((c = waitKey()) > 255);
    }

    Mat hsv;
    cvtColor( denoised, hsv, CV_BGR2HSV );

    vector<Mat> hsv_planes;
    split(hsv, hsv_planes);

    Mat hue = hsv_planes[0];
    {
        imshow(wndname, hue);
        int c;
        while((c = waitKey()) > 255);
    }
    Mat lightness = hsv_planes[2];

    Mat saturation = hsv_planes[1];
    cout << "Saturation size: " << saturation.size() << " type: " << getImageType(saturation.type()) << endl;

    {
        imshow(wndname, saturation);
        int c;
        while((c = waitKey()) > 255);
    }

    Mat colorOnly;
    inRange(saturation, 25, 255, colorOnly);

    {
      imshow(wndname, colorOnly);
      int c;
      //while((c = waitKey()) > 255);
    }

    Mat colorOnlyClosed;
    Mat kernel(3,3,CV_8U,Scalar(1));
    morphologyEx(colorOnly, colorOnlyClosed, MORPH_CLOSE, kernel);
    {
      imshow(wndname, colorOnlyClosed);
      int c;
      while((c = waitKey()) > 255);
    }

    Mat colorOnlyOpened;
    morphologyEx(colorOnlyClosed, colorOnlyOpened, MORPH_OPEN, kernel);
    {
      imshow(wndname, colorOnlyOpened);
      int c;
      //while((c = waitKey()) > 255);
    }

#if 0
    Mat sure_bg;
    dilate(colorOnlyClosed, sure_bg, kernel ,Point(-1,-1), 3);
    {
      imshow(wndname, sure_bg);
      int c;
      while((c = waitKey()) > 255);
    }
#else
    Mat sure_bg(colorOnlyClosed);
#endif

    Mat sure_bg_neg;
    bitwise_not(sure_bg, sure_bg_neg);
    {
      imshow(wndname, sure_bg_neg);
      int c;
      //while((c = waitKey()) > 255);
    }

    cout << "ColorOnlyClosed size: " << colorOnlyClosed.size() << " type: " << getImageType(colorOnlyClosed.type()) << endl;

    Mat dist;
    distanceTransform(colorOnlyClosed, dist, CV_DIST_L2, 3);
    cout << "Unnormalized dist size: " << dist.size() << " type: " << getImageType(dist.type()) << endl;

    normalize(dist, dist, 0, 255, NORM_MINMAX, CV_8UC1);
    cout << "Dist size: " << dist.size() << " type: " << getImageType(dist.type()) << endl;

    {
      imshow(wndname, dist);
      int c;
      while((c = waitKey()) > 255);
    }

    Mat sure_fg;
    {
      //double maxval = *std::max_element(dist.begin<double>(),dist.end<double>());
      double minVal, maxVal;
      minMaxLoc( dist, &minVal, &maxVal);
      cout << "Max value is " << maxVal << endl;

      threshold(dist, sure_fg, 0.4 * maxVal, 255, CV_THRESH_BINARY);

      cout << "Sure_fg size: " << sure_fg.size() << " type: " << getImageType(sure_fg.type()) << endl;
      {
        imshow(wndname, sure_fg);
        int c;
        //while((c = waitKey()) > 255);
      }
    }


    // Find total markers
    std::vector<std::vector<Point> > contours;
    findContours(sure_fg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    int ncomp = contours.size();
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    for (int i = 0; i < ncomp; i++)
        drawContours(markers, contours, i, Scalar::all(i+1), -1);

    markers.setTo(Scalar(ncomp + 1), sure_bg_neg);

    cout << "markers size: " << markers.size() << " type: " << getImageType(markers.type()) << endl;

    Mat normmarkers;
    normalize(markers, normmarkers, 0, 255, NORM_MINMAX, CV_8UC1);
    {
      imshow(wndname, normmarkers);
      int c;
      while((c = waitKey()) > 255);
    }

    Mat prewatershed;
    cvtColor(colorOnlyClosed, prewatershed, CV_GRAY2BGR, 3);
    watershed(hsv, markers);
    normalize(markers, normmarkers, 0, 255, NORM_MINMAX, CV_8UC1);
    {
      imshow(wndname, normmarkers);
      int c;
      while((c = waitKey()) > 255);
    }

    std::vector<std::vector<Point> > componentContours;
    for(int i = 0; i < ncomp; i++)
    {
        Mat singlecomponent;
        inRange(markers, i, i, singlecomponent);
        findContours(singlecomponent, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        if(contours.size() != 0)
        {
            if(contours.size() > 1)
            {
                cout << "More than one (" << contours.size() << ") external contour found for component " << i << endl;
            }
#if 1
            componentContours.push_back(contours[0]);
#else
            vector<Point> approx;
            approxPolyDP(Mat(contours[0]), approx, arcLength(Mat(contours[0]), true)*0.01, true);
            cout << "Approximated contour consists of " << approx.size() << "points" << endl;
            componentContours.push_back(approx);
#endif
        }
    }


    std::vector<Mat> templates;

    for(int selected_contour = 0; selected_contour < componentContours.size(); selected_contour++)
    //int selected_contour = 19;
    {

      Mat eachcontour = Mat::zeros(dist.size(), CV_8UC1);

      drawContours(eachcontour, componentContours, selected_contour, Scalar(255), -1);
      {
        imshow(wndname, eachcontour);
        int c;
        //while((c = waitKey()) > 255);
      }

      Point2f center;
      float radius;
      minEnclosingCircle(componentContours[selected_contour], center, radius);
      //cout << "center is " << center << " radius is " << radius << endl;

      Size size(radius * 2, radius * 2);
      Mat singlecontour(size, CV_8UC1);
      Mat imagecutout;
      getRectSubPix(eachcontour, size, center, singlecontour);
      getRectSubPix(image, size, center, imagecutout);
      Point2f newcenter(size.width / 2.0, size.height / 2.0);
      //circle(singlecontour, newcenter, radius, Scalar(255,255,255));

      double overallMax = 0;
      Point overallMaxLoc;
      int overallMaxAngle;
      int overallMaxKernelSize;

      Mat rotatedcontour;

      for(int angle = -16; angle <= 16; angle += 1)
      //int angle = 0;
      {
        Mat rotation = getRotationMatrix2D(newcenter, angle, 1.0);
        warpAffine(singlecontour, rotatedcontour, rotation, singlecontour.size(), INTER_CUBIC);

        {
          imshow(wndname, rotatedcontour);
          int c;
          //while((c = waitKey()) > 255);
        }

        Mat offsetcontour;
        subtract(rotatedcontour, 160, offsetcontour, noArray(), CV_32F);


        for(int kernel_size = 50; kernel_size < 70; kernel_size += 1)
        //int kernel_size = 60;
        {
#if 0
          Mat horizkernel;
          Mat horiz;
          horizkernel = Mat::ones( 1, kernel_size, CV_32F );
          filter2D(offsetcontour, horiz, CV_32F, horizkernel, Point( 0, 0 ), 0, BORDER_DEFAULT );

          Mat vertkernel;
          Mat vert;
          vertkernel = Mat::ones( kernel_size, 1, CV_32F );
          filter2D(horiz, vert, -1, vertkernel, Point( 0, 0 ), 0, BORDER_DEFAULT );

          Mat normvert;
          normalize(vert, normvert, 0, 255, NORM_MINMAX, CV_8UC1);

          //cout << "vert: " << vert << endl;

          {
            imshow(wndname, normvert);
            int c;
            //while((c = waitKey()) > 255);
          }
          Mat temp = vert;
#else

          Mat boxfiltered;
          boxFilter(offsetcontour, boxfiltered, CV_32F, Size(kernel_size, kernel_size), Point(0,0), false);

          //cout << "boxfiltered: " << boxfiltered << endl;

          Mat normbox;
          normalize(boxfiltered, normbox, 0, 255, NORM_MINMAX, CV_8UC1);
          {
            imshow(wndname, normbox);
            int c;
            //while((c = waitKey()) > 255);
          }

          Mat temp = boxfiltered;
#endif

          if(kernel_size < temp.size().width && kernel_size < temp.size().height)
          {
            Mat mask = Mat::zeros(temp.size(), CV_8UC1);
            mask(Rect(0,0,temp.size().width - kernel_size + 1, temp.size().height - kernel_size + 1)) = 1;

            double minVal, maxVal;
            Point maxLoc;
            minMaxLoc(temp, &minVal, &maxVal, 0, &maxLoc, mask);
            //cout << "max val at rotation " << angle << " with kernel size " << kernel_size << " is " << maxVal << " at pos " << maxLoc << endl;
            if(maxVal > overallMax)
            {
              overallMax = maxVal;
              overallMaxLoc = maxLoc;
              overallMaxAngle = angle;
              overallMaxKernelSize = kernel_size;
            }
          }
        }
      }


      Mat resizedimage;
      Mat rotation = getRotationMatrix2D(newcenter, overallMaxAngle, 1.0);
      Rect roi(overallMaxLoc.x, overallMaxLoc.y, overallMaxKernelSize, overallMaxKernelSize);

      warpAffine(imagecutout, rotatedcontour, rotation, imagecutout.size(), INTER_CUBIC);
      if(overallMax > 0)
      {
        cout << "Overall max match was " << overallMax << " at rotation " << overallMaxAngle << " with kernel size " << overallMaxKernelSize << " at location " << overallMaxLoc << endl;
        templates.push_back(rotatedcontour(roi));
      }

#if 0
      rectangle(rotatedcontour, overallMaxLoc, overallMaxLoc + Point(overallMaxKernelSize, overallMaxKernelSize), 0);
      resize(rotatedcontour, resizedimage, Size(0, 0), 2, 2, INTER_NEAREST);

      {
        imshow(wndname, resizedimage);
        int c;
        //while((c = waitKey()) > 255);
      }

      Mat resizedmask1;
      Mat resizedmask3;

      warpAffine(singlecontour, rotatedcontour, rotation, imagecutout.size(), INTER_CUBIC);
      rectangle(rotatedcontour, overallMaxLoc, overallMaxLoc + Point(overallMaxKernelSize, overallMaxKernelSize), 0);
      resize(rotatedcontour, resizedmask1, Size(0, 0), 2, 2, INTER_NEAREST);
      cvtColor(resizedmask1, resizedmask3, CV_GRAY2BGR, 3);

      {
        imshow(wndname, resizedmask3);
        int c;
        //while((c = waitKey()) > 255);
      }

      cout << "resizedimage size: " << resizedimage.size() << " type: " << getImageType(resizedimage.type()) << endl;
      cout << "resizedmask3 size: " << resizedmask3.size() << " type: " << getImageType(resizedmask3.type()) << endl;

      Mat blended;
      float alpha = 0.7;
      addWeighted(resizedimage, alpha, resizedmask3, 1.0 - alpha, 0.0, blended);
      {
        imshow(wndname, blended);
        int c;
        //while((c = waitKey()) > 255);
      }
#endif
    }


#if 1
    Mat hsvimage;
    cvtColor(image, hsvimage, CV_BGR2HSV);
    vector<Mat> hsvimageplanes;
    split(hsvimage, hsvimageplanes);

    int diagonal = (int)ceil(sqrt(denoised.cols * denoised.cols + denoised.rows * denoised.rows));

    int half_height = denoised.rows / 2;
    double half_diag = diagonal / 2.0;

    int maxangle = 16;
    double maxangle_rad = M_PI * maxangle / 180;

    double diagangle_rad = asin(half_height / half_diag);
    int max_half_height = ceil(sin(diagangle_rad + maxangle_rad) * half_diag);
    int max_half_width = ceil(cos(diagangle_rad - maxangle_rad) * half_diag);

    int x_offset = max_half_width - denoised.cols / 2;
    int y_offset = max_half_height - denoised.rows / 2;

    Mat largeimage(max_half_height * 2,
                   max_half_width * 2,
                   denoised.type());
    Rect destrect(x_offset,
                  y_offset,
                  denoised.cols,
                  denoised.rows);


    Mat dest = largeimage(destrect);
    image.copyTo(dest);
    {
      imshow(wndname, largeimage);
      int c;
      //while((c = waitKey()) > 255);
    }

    Point center(largeimage.cols / 2, largeimage.rows / 2);
    Mat rotatedimage;


    Mat processedimage = largeimage;

    vector<Point> templateLocations;

    //for(int templateidx = 0; templateidx < templates.size(); templateidx++)
    int templateidx = 1;
    {
      double overallMinDiff = 1000000;
      Point overallMinLoc;

      Mat templateimage = templates[templateidx];

      {
        imshow(wndname, templateimage);
        int c;
        while((c = waitKey()) > 255);
      }

      for(int angle = -16; angle <= 16; angle += 4)
      //int angle = 1;
      {
        Mat rotation = getRotationMatrix2D(center, angle, 1.0);
        warpAffine(processedimage, rotatedimage, rotation, processedimage.size(), INTER_CUBIC);


        Mat hsv;
        cvtColor( rotatedimage, hsv, CV_BGR2HSV );

        vector<Mat> hsv_planes;
        split(hsv, hsv_planes);

        Mat saturation = hsv_planes[1];
        Mat onlyColor;
        inRange(saturation, 25, 255, onlyColor);
        Mat onlyColorDilated;
        dilate(onlyColor, onlyColorDilated, 2);
        {
          imshow(wndname, onlyColorDilated);
          int c;
          //while((c = waitKey()) > 255);
        }

        Mat diffimage(rotatedimage.size(), CV_32F);
        double mindiff = 1000000;
        int minx, miny;
        int count = 0;
        for(int y = 0; y < (rotatedimage.rows - templateimage.rows + 1); y++)
        {
          for(int x = 0; x < (rotatedimage.cols - templateimage.cols + 1); x++)
          {
            if(onlyColorDilated.at<uchar>(y, x) > 128)
            {
              Rect roirect(x, y, templateimage.cols, templateimage.rows);
              Mat roi = rotatedimage(roirect);
              double diff = norm(roi, templateimage, NORM_L1);
              diffimage.at<float>(y,x) = diff;
              if(diff < mindiff)
              {
                //cout << "previous min was " << mindiff << endl;
                mindiff = diff;
                minx = x;
                miny = y;
              }
              count++;
            }
          }
        }
        cout << "positions checked: " << count << endl;

        cout << "minimum of " << mindiff << " at (" << minx << "," << miny << ")" << endl;


        if(mindiff < overallMinDiff)
        {
          overallMinDiff = mindiff;

          Point minpos(minx, miny);
          Point result;
          Mat M = getRotationMatrix2D(center, -angle, 1.0);
          result.x = M.at<double>(0,0)*minpos.x + M.at<double>(0,1)*minpos.y + M.at<double>(0,2);
          result.y = M.at<double>(1,0)*minpos.x + M.at<double>(1,1)*minpos.y + M.at<double>(1,2);

          result.x -= x_offset;
          result.y -= y_offset;

          overallMinLoc = result;
        }

        Mat diffimagenorm;
        normalize(diffimage, diffimagenorm, 0, 255, NORM_MINMAX, CV_8UC1);
        {
          imshow(wndname, diffimagenorm);
          int c;
          //while((c = waitKey()) > 255);
        }
      }
      templateLocations.push_back(overallMinLoc);
    }

    Mat markedtemplates(image.size(), image.type());
    image.copyTo(markedtemplates);
    for(int i = 0; i < templateLocations.size(); i++)
    {
      circle(markedtemplates, templateLocations[i], 3, Scalar(255,0,0), 2);
    }
    {
      imshow(wndname, markedtemplates);
      int c;
      while((c = waitKey()) > 255);
    }



#else

    Mat templ = templates[0];

    ORB orb(500, 1.0f, 1, 5, 0, 2, ORB::HARRIS_SCORE, 5);
    vector<KeyPoint> keypointsimg, keypointstempl;
    Mat descriptorsimg, descriptorstempl;
    orb(denoised, Mat(), keypointsimg, descriptorsimg);
    orb(templ, Mat(), keypointstempl, descriptorstempl);

    {
      imshow(wndname, descriptorsimg);
      int c;
      while((c = waitKey()) > 255);
    }

    Mat matchimage;
    vector<DMatch> matches;

#if 1
    BFMatcher matcher(NORM_HAMMING, true);
    matcher.match(descriptorstempl, descriptorsimg, matches);
#else
    FlannBasedMatcher matcher;
    matcher.match( descriptorstempl, descriptorsimg, matches);

#endif


    drawMatches(templ, keypointstempl, denoised, keypointsimg, matches, matchimage);

    {
      imshow(wndname, matchimage);
      int c;
      while((c = waitKey()) > 255);
    }

#endif

}


int main(int /*argc*/, char** /*argv*/)
{
    static const char* names[] = { "scrumboardphoto.jpg", 0 };
    help();
    namedWindow( wndname, 1 );
    moveWindow(wndname, 100, 100);

    for( int i = 0; names[i] != 0; i++ )
    {
        Mat image = imread(names[i], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findSquares(image);
    }

    return 0;
}
