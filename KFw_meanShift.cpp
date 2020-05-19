#include <iostream>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


class trackObj{
    VideoCapture cap;
    Mat frame, res;

    // Kalman Filter variables
    int stateSize;
    int measSize;
    int contrSize;
    unsigned int type;

    KalmanFilter kf;
    Mat state, meas;
    double ticks;
    bool found;
    int notFoundCount;
    Rect bBox;
    Rect predRect;
    
    int idx;
    
    // Meanshift Variables
    Rect track_window;
    Mat roi, hsv_roi, mask, roi_hist, hsv, dst;
    
    public:
    trackObj(string path);
    void current_frame();
    void meanShift_trackWindow();
    void Kalman_filter();
};

trackObj::trackObj(string path){
    // Open video
    found = false;
    cap.open(path);
    if(!cap.isOpened()){
        cout<<"Error opening video stream or file"<<endl;
    }

    cap>>frame;

    // Initializing variables for meanshift
    track_window.x = 305;
    track_window.y = 225;
    track_window.width = 60;
    track_window.height = 60;

    roi = frame(track_window);
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);

    float range_[]  = {0, 180};
    const float* range[] = {range_};

    int histSize[] = {180};
    int channels[] = {0};
    
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    // Intializing variables for Kalman Filters
    ticks = 0;
    contrSize = 0;
    measSize = 4;
    stateSize = 6;
    found = false;
    type = CV_32F;
    notFoundCount = 0;

    kf.init(stateSize, measSize, contrSize, type);
    
    state = Mat(stateSize, 1, type);
    meas = Mat(measSize, 1, type);

    setIdentity(kf.transitionMatrix);

    kf.measurementMatrix = Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;
    setIdentity(kf.measurementNoiseCov, Scalar(1e-3));

}

void trackObj :: current_frame(){

    while(1){

        cap>>frame;

        if(frame.empty())
            break;

        meanShift_trackWindow();
        Kalman_filter();

        rectangle(res, track_window, CV_RGB(0,255,0), 2);
        rectangle(res, predRect, CV_RGB(255,0,0), 2);

        imshow("img2", res);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}

void trackObj :: meanShift_trackWindow(){

    float range_[]  = {0, 180};
    const float* range[] = {range_};

    int histSize[] = {180};
    int channels[] = {0};

    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

    meanShift(dst, track_window, term_crit);
    


}

void trackObj :: Kalman_filter(){
    double precTick = ticks;
    ticks = (double)getTickCount();
    
    double dT = (ticks-precTick)/cv::getTickFrequency();

    frame.copyTo(res);

    // if(found)
    kf.transitionMatrix.at<float>(2) = dT;
    kf.transitionMatrix.at<float>(9) = dT;

    predRect.width = state.at<float>(4);
    predRect.height = state.at<float>(5);
    predRect.x = state.at<float>(0) - predRect.width/2;
    predRect.y = state.at<float>(1) - predRect.height/2;

    Point center;
    center.x = state.at<float>(0);
    center.y = state.at<float>(1);
    
    circle(res, center, 2, CV_RGB(255,0,0), -1);
    

    meas.at<float>(0) = track_window.x + track_window.width/2;
    meas.at<float>(1) = track_window.y + track_window.height/2;
    meas.at<float>(2) = (float)track_window.width;
    meas.at<float>(3) = (float)track_window.height;

    if(!found){
        kf.errorCovPre.at<float>(0) = 1;
        kf.errorCovPre.at<float>(7) = 1;
        kf.errorCovPre.at<float>(14) = 1;
        kf.errorCovPre.at<float>(21) = 1;
        kf.errorCovPre.at<float>(28) = 1;
        kf.errorCovPre.at<float>(35) = 1;

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(1) = meas.at<float>(1);
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = meas.at<float>(2);
        state.at<float>(5) = meas.at<float>(3);

        kf.statePost = state;
        found = true;

    }
    else{
        kf.correct(meas);
    }


}
int main(){
    
    
    string path = "/home/joshua/FrictionFinger_videos/2.mp4";
    trackObj t(path);
    t.current_frame();


}