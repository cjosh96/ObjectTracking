#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
// #include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>


using namespace std;

#define MIN_H_BLUE 150
#define MAX_H_BLUE 250

int main(){

    cv::VideoCapture cap("/home/joshua/FrictionFinger_videos/1.mp4");
    // cv::VideoCapture cap("/home/joshua/FrictionFinger_videos/Simulation_5.mp4"); 
    cv::Mat frame;
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);          // state variables (position, velocity, window size)
    cv::Mat meas(measSize, 1, type);            // measure vector (observed position, window size)
    
    cv::setIdentity(kf.transitionMatrix);

    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
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

    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-3));


    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    cap>>frame;

    cv::Mat roi, hsv_roi;
    // cv::Rect track_window(510, 230, 20, 20);
    // roi = frame(track_window);
    // cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
    // rectangle(frame, track_window, cv::Scalar(0,255,0), 4, 8, 0);
    // cv::imshow("RoI", frame);
    // cv::waitKey(0);
    int idx = 0;

    char ch = 0;

    double ticks = 0;
    bool found = false;

    int notFoundCount = 0;

    while(1){
        double precTick = ticks;
        ticks = (double) cv::getTickCount();
        
        double dT = (ticks-precTick)/cv::getTickFrequency();
        cap>>frame;
        cv::Mat res;
        frame.copyTo(res);

        if(found){
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;

            cout<<"dT "<<endl<<state<<endl;

            cv::Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width/2;
            predRect.y = state.at<float>(1) - predRect.height/2;

            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cv::circle(res, center, 2, CV_RGB(255,0,0), -1);

            cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);
        }

        cv::Mat blur;
        cv::GaussianBlur(frame, blur, cv::Size(5,5), 3.0, 3.0);

        cv::Mat frmHsv;
        cv::cvtColor(blur, frmHsv, CV_BGR2HSV);

        cv::Mat rangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        // cv::inRange(frmHsv, cv::Scalar(80,30,50), cv::Scalar(120,255,255), rangeRes); // For simulation video
        // cv::inRange(frmHsv, cv::Scalar(105,30,150), cv::Scalar(115,50,200), rangeRes); // For actual video
        cv::inRange(frmHsv, cv::Scalar(20,20,20), cv::Scalar(70,100,100), rangeRes); 
        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1,-1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1,-1), 2);

        cv::imshow("threshold", rangeRes);
        vector<vector<cv::Point> > contours;

        cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        vector<vector<cv::Point> > balls;
        vector<cv::Rect> ballsBox;

        for(size_t i = 0; i< contours.size(); i++){
            cv::Rect bBox;
            bBox = cv::boundingRect(contours[i]);

            float ratio = (float) bBox.width/(float) bBox.height;

            if(ratio > 1.0f)
                ratio = 1.0f/ratio;
            
            if(ratio > 0.5 &&bBox.area() >=100){
                balls.push_back(contours[i]);
                ballsBox.push_back(bBox);
            }
        }

        cout<<"Object Found :"<<ballsBox.size()<<endl;

        for(size_t i = 0; i<balls.size(); i++){
            cv::drawContours(res, balls, i, CV_RGB(20,150,20),1);
            cv::rectangle(res, ballsBox[i], CV_RGB(0,255,0), 2);

            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width/2;
            center.y = ballsBox[i].y + ballsBox[i].height/2;
            cv::circle(res, center, 2, CV_RGB(20,150,20), -1);

            // stringstream sstr;
            // sstr<<"("<<center.x<<","<<center.y<<")";
            // cv::putText(res, sstr.str(), cv::Point(center.x+3, center.y-3), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20),2);

        }

        if(balls.size() ==0){
            notFoundCount++;
            cout<<"notFoundCount:"<<notFoundCount<<endl;
            if(notFoundCount >= 100){
                found = false;
            }
        }
        else{
            notFoundCount = 0;

            meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width/2;
            meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height/2;
            meas.at<float>(2) = (float)ballsBox[0].width;
            meas.at<float>(3) = (float)ballsBox[0].height;

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

            cout<<"Measure matrix:"<<endl<<meas<<endl;
        }
        cv::imshow("Tracking", res);
        ch = cv::waitKey(1);
      
    }
}