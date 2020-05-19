#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <vector>


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
    
    int idx;

    //Optical Flow Variables
    vector<Point2f> p0, p1;
    Mat mask, old_gray;



    public:
    trackObj();
    void current_frame();
    void Kalman_filter_1();
    void Opt_flow();
    

};

trackObj::trackObj(){
    
    // Video
    cap.open("/home/joshua/FrictionFinger_videos/video_Hex.mp4");

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
    } 

    // Initializing variables for Kalman Filter
    ticks = 0;
    contrSize = 0;
    measSize = 4;
    stateSize = 6;
    found  = false;
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
    }
}

void trackObj :: Kalman_filter_1(){

    double precTick = ticks;
    ticks = (double) getTickCount();        
    double dT = (ticks-precTick)/getTickFrequency();
    frame.copyTo(res);

    if(found){
        kf.transitionMatrix.at<float>(2) = dT;
        kf.transitionMatrix.at<float>(9) = dT;

        cout<<"dT "<<endl<<state<<endl;

        Rect predRect;
        predRect.width = state.at<float>(4);
        predRect.height = state.at<float>(5);
        predRect.x = state.at<float>(0) - predRect.width/2;
        predRect.y = state.at<float>(1) - predRect.height/2;

        Point center;
        center.x = state.at<float>(0);
        center.y = state.at<float>(1);
        rectangle(res, predRect, CV_RGB(255,0,0), 2);
    
    }
    Mat blur;
    GaussianBlur(frame, blur, Size(5,5), 3.0, 3.0);

    Mat frmHsv;
    cvtColor(blur, frmHsv, CV_BGR2HSV);

    Mat rangeRes = Mat::zeros(frame.size(), CV_8UC1);

    inRange(frmHsv, Scalar(105,30,150), Scalar(115,50,200), rangeRes); 
    erode(rangeRes, rangeRes, Mat(), Point(-1,-1), 2);
    dilate(rangeRes, rangeRes, Mat(), Point(-1,-1), 2);

    vector<vector<Point> > contours;

    findContours(rangeRes, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    vector<vector<Point> > balls;
    vector<Rect> ballsBox;


    for(size_t i = 0; i< contours.size(); i++){

        bBox = boundingRect(contours[i]);

        float ratio = (float) bBox.width/(float) bBox.height;

        if(ratio > 1.0f)
            ratio = 1.0f/ratio;
            
        if(ratio > 0.5 &&bBox.area() >=100){
            balls.push_back(contours[i]);
            ballsBox.push_back(bBox);
        }
    }


    for(size_t i = 0; i<balls.size(); i++){
        drawContours(res, balls, i, CV_RGB(20,150,20),1);
        rectangle(res, ballsBox[i], CV_RGB(0,255,0), 2);

        Point center;
        center.x = ballsBox[i].x + ballsBox[i].width/2;
        center.y = ballsBox[i].y + ballsBox[i].height/2;
    }

    if(balls.size() ==0){
        notFoundCount++;
        if(notFoundCount >= 100)
            found = false; 
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
        else
            kf.correct(meas);
    }
}

void trackObj :: Opt_flow(){


    
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    if(idx == 0){
        Mat ROI;
        ROI = frame_gray(bBox);
        goodFeaturesToTrack(frame_gray, p0, 15, 0.01, 10, Mat(), 3, false, 0.04);
    }
    else{
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);

        vector<Point2f> good_new;

        for(int i = 0; i<p0.size(); i++){
            if(status[i] == 1){
                good_new.push_back(p1[i]);
            }
            line(mask,p1[i], p0[i], Scalar(0,0,255), 2);
            circle(frame, p1[i], 5, Scalar(0,255,0), -1);
        }
        Mat img;
        old_gray = frame_gray.clone();
        p0 = good_new;
        add(frame, mask, img);
        
    }   
}