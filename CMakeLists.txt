cmake_minimum_required(VERSION 2.8)
project( KalmanFilter_trackingObj )
find_package( OpenCV REQUIRED )



add_executable( KalmanFilter_trackingObj KalmanFilter_trackingObj.cpp )
add_executable( test_kf test.cpp  )
add_executable( Optical_flow Optical_flow.cpp  )
add_executable( KFw_meanShift KFw_meanShift.cpp  )
add_executable( VFF_kinematics VFF_kinematics.cpp )

target_link_libraries( KalmanFilter_trackingObj ${OpenCV_LIBS} )
target_link_libraries( test_kf ${OpenCV_LIBS} )
target_link_libraries( Optical_flow ${OpenCV_LIBS} )
target_link_libraries( KFw_meanShift ${OpenCV_LIBS} )
target_link_libraries( VFF_kinematics symbolicc++)
