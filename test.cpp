#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>


using namespace std;

int main(){
    cv::Mat frame;
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;
    unsigned int type = CV_32F;


    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

    cv::setIdentity(kf.transitionMatrix);
    cout<<kf.measurementNoiseCov;
}