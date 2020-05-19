#include <iostream>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

int main(){
    VideoCapture cap("/home/joshua/FrictionFinger_videos/2.mp4");
    if(!cap.isOpened()){
        cout<<"error opening file";
        return 0;
    }
    Mat frame, roi, hsv_roi, mask;
    cap>>frame;

    Rect track_window(305, 225, 60, 60);
    roi = frame(track_window);
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);

    float range_[] = {0, 180};
    const float* range[] = {range_};

    Mat roi_hist;
    int histSize[] = {180};
    int channels[] = {0};
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);

    while(true){
        Mat hsv, dst;
        cap>>frame;
        
        if(frame.empty())
            break;
        
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

        meanShift(dst, track_window, term_crit);

        // Draw it on image
        rectangle(frame, track_window, 255, 2);
        imshow("img2", frame);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;

    }    




}