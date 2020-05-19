#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <vector>

using namespace std;

int main(){
    cv::Mat old_frame, old_gray;
    cv::VideoCapture cap("/home/joshua/FrictionFinger_videos/video_Hex.mp4");
    cap>>old_frame;
    cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
    vector<cv::Point2f> p0, p1;
    // void goodFeaturesToTrack(InputArray image, OutputArray corners, int maxCorners, 
    //                          double qualityLevel, double minDistance, InputArray mask=noArray(), 
    //                          int blockSize=3, bool useHarrisDetector=false, double k=0.04)
    
    goodFeaturesToTrack(old_gray, p0, 15, 0.01, 10, cv::Mat(), 3, false, 0.04);
    while(true){
        cv::Mat frame, frame_gray;
        cap>>frame;
        if(frame.empty())
            break;

        cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        vector<uchar> status;
        vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, cv::Size(15,15), 2, criteria);

        // Create a mask image for drawing purposes
        cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

        vector<cv::Point2f> good_new;

        for(int i = 0; i<p0.size(); i++){
            if(status[i] == 1){
                good_new.push_back(p1[i]);
            }
            line(mask,p1[i], p0[i], cv::Scalar(0,0,255), 2);
            circle(frame, p1[i], 5, cv::Scalar(0,255,0), -1);
        }
        cv::Mat img;
        add(frame, mask, img);

        imshow("Frame", img);
        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        old_gray = frame_gray.clone();
        p0 = good_new;
        
    }
}