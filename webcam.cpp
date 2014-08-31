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

const char* wndname = "Webcam";

int main(int /*argc*/, char** /*argv*/)
{
    namedWindow( wndname, 1 );
    moveWindow(wndname, 100, 100);

    Mat image = imread("webcam.jpg", 1);

    {
      imshow(wndname, image);
      int c;
      while((c = waitKey()) > 255);
    }

    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    int average_bgr[3] = {84, 135, 137};

    for(int i = 0; i < 3; i++)
    {
      int thresh = average_bgr[i];
      Mat mask;
      Mat plane = bgr_planes[i];
      {
          cout << "Showing plane" << endl;
          imshow(wndname, plane);
          int c;
          while((c = waitKey()) > 255);
      }

      threshold(plane, mask, thresh, 255, THRESH_BINARY);
      {
          cout << "Showing mask" << endl;
          imshow(wndname, mask);
          int c;
          //while((c = waitKey()) > 255);
      }

      Mat highvalues;
      highvalues = (plane - thresh) * (128.0 / (255 - thresh)) + 128;

      Mat highvalues_masked;
      bitwise_and(highvalues, mask, highvalues_masked);
      {
          cout << "Showing scaled high values" << endl;
          imshow(wndname, highvalues_masked);
          int c;
          //while((c = waitKey()) > 255);
      }



      mask = 255 - mask;
      Mat lowvalues;
      bitwise_and(plane, mask, lowvalues);
      {
          cout << "Showing low values" << endl;
          imshow(wndname, lowvalues);
          int c;
          //while((c = waitKey()) > 255);
      }

      lowvalues = lowvalues * 128.0 / thresh;
      {
          cout << "Showing scaled low values" << endl;
          imshow(wndname, lowvalues);
          int c;
          //while((c = waitKey()) > 255);
      }

      bgr_planes[i] = lowvalues + highvalues_masked;
      {
          cout << "Showing scaled plane" << endl;
          imshow(wndname, bgr_planes[i]);
          int c;
          while((c = waitKey()) > 255);
      }
    }

    Mat correctedimage;
    merge(bgr_planes, correctedimage);
    {
        cout << "Showing corrected image" << endl;
        imshow(wndname, correctedimage);
        int c;
        while((c = waitKey()) > 255);
    }

    Mat hsv;
    cvtColor( correctedimage, hsv, CV_BGR2HSV );

    vector<Mat> hsv_planes;
    split(hsv, hsv_planes);

    Mat saturation = hsv_planes[1];
    {
        cout << "Showing saturation of corrected image" << endl;
        imshow(wndname, saturation);
        int c;
        while((c = waitKey()) > 255);
    }

    return 0;
}
