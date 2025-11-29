#include "../tasks/detector.hpp"
#include "../tools/img_tools.hpp"
#include "fmt/core.h"
#include <chrono>
#include <cmath>
#include "../506/MVS/include/MvCameraControl.h"  // SDK 核心头文件
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// clang-format off
//  相机内参
static const cv::Mat camera_matrix =
    (cv::Mat_<double>(3, 3) <<  1286.307063384126 , 0                  , 645.34450819155256, 
                                0                 , 1288.1400736562441 , 483.6163720308021 , 
                                0                 , 0                  , 1                   );
// 畸变系数
static const cv::Mat distort_coeffs =
    (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
// clang-format on

static const double LIGHTBAR_LENGTH = 0.056; // 灯条长度    单位：米
static const double ARMOR_WIDTH = 0.135;     // 装甲板宽度  单位：米




static const std::vector<cv::Point3f> object_points {
    {       -ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2  , 0 },  // 点 1
    {        ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2  , 0 },  // 点 2
    {        ARMOR_WIDTH / 2,  LIGHTBAR_LENGTH / 2  , 0 },  // 点 3
    {       -ARMOR_WIDTH / 2,  LIGHTBAR_LENGTH / 2  , 0 }   // 点 4
};


int main(int argc, char *argv[])
{
    auto_aim::Detector detector;

    //cv::VideoCapture cap("video.avi");
    

    VideoCapture cap(1);
    if (!cap.isOpened()) {
        cerr << "相机打开失败！" << endl;
        return -1;
    }

    cv::Mat img;

    //Pose6D last_pose = {0,0,0,0,0,0};
    //auto last_time = chrono::steady_clock::now();




    while (true)
    {
        cap >> img;
        if (img.empty()) // 读取失败 或 视频结尾
            break;

        auto armors = detector.detect(img);

        if (!armors.empty())
        {
            auto armor = armors.front();           // 如果识别到了大于等于一个装甲板，则取出第一个装甲板来处理
            tools::draw_points(img, armor.points); // 绘制装甲板



            std::vector<cv::Point2f> img_points{ armor.left.top,armor.right.top,armor.right.bottom,armor.left.bottom};
  



            cv::Mat rvec, tvec;

            cv::solvePnP(object_points,img_points,camera_matrix,distort_coeffs,rvec,tvec);
       



      
            tools::draw_text(img, fmt::format("tvec:  x{: .2f} y{: .2f} z{: .2f}", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)), cv::Point2f(10, 60), 1.7, cv::Scalar(0, 255, 255), 3);
            tools::draw_text(img, fmt::format("rvec:  x{: .2f} y{: .2f} z{: .2f}", rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)), cv::Point2f(10, 120), 1.7, cv::Scalar(0, 255, 255), 3);
           


            cv::Mat rmat;
            cv::Rodrigues(rvec,rmat);
            double yaw  = std::atan2(rmat.at<double>(0,2),rmat.at<double>(2,2));
            double pitch= std::asin(-rmat.at<double>(1,2));
            double roll = std::atan2(rmat.at<double>(1,0),rmat.at<double>(1, 1));
            tools::draw_text(img, fmt::format("euler angles:  yaw{: .2f} pitch{: .2f} roll{: .2f}", yaw, pitch, roll), cv::Point2f(10, 180), 1.7, cv::Scalar(0, 255, 255), 3);


        }

        cv::imshow("press q to quit", img);

        if (cv::waitKey(20) == 'q')
            break;
    }

    cv::destroyAllWindows();
    return 0;
}