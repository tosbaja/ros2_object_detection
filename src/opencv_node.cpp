#include <iostream>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <image_transport/image_transport.hpp>

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>


using std::placeholders::_1;

class DnnParams
{
  public:

    DnnParams(){
      std::ifstream ifs(std::string(file_path + "object_detection_classes_coco.txt").c_str());
          while (getline(ifs, line))
        {
          class_names.push_back(line);
        } 
      net = cv::dnn::readNet(file_path + "frozen_inference_graph.pb", 
      file_path + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt","TensorFlow");
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    float min_confidence_score = 0.5;
    std::vector<std::string> class_names;
    std::string file_path = "/home/dev/opencv_ws/src/opencv_package/config/";
    cv::dnn::Net net;
    std::string line;

    auto process(cv::Mat image){ 
      int image_height = image.cols;
      int image_width = image.rows;

      auto start = cv::getTickCount();

      // Create a blob from the image
      cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5),
                              true, false);

      
      // Set the blob to be input to the neural network
      this->net.setInput(blob);

      // Forward pass of the blob through the neural network to get the predictions
      cv::Mat output = this->net.forward();

      auto end = cv::getTickCount();

       // Matrix with all the detections
       cv::Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());
       
       // Run through all the predictions
       for (int i = 0; i < results.rows; i++){
           int class_id = int(results.at<float>(i, 1));
           float confidence = results.at<float>(i, 2);
  
           // Check if the detection is over the min threshold and then draw bbox
           if (confidence > min_confidence_score){
               int bboxX = int(results.at<float>(i, 3) * image.cols);
               int bboxY = int(results.at<float>(i, 4) * image.rows);
               int bboxWidth = int(results.at<float>(i, 5) * image.cols - bboxX);
               int bboxHeight = int(results.at<float>(i, 6) * image.rows - bboxY);
               cv::rectangle(image, cv::Point(bboxX, bboxY), cv::Point(bboxX + bboxWidth, bboxY + bboxHeight), cv::Scalar(0,0,255), 2);
               std::string class_name = class_names[class_id-1];
               putText(image, class_name + " " + std::to_string(int(confidence*100)) + "%", cv::Point(bboxX, bboxY - 10), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0,255,0), 2);
           }
       }
       

      auto totalTime = (end - start) / cv::getTickFrequency();
        

      putText(image, "FPS: " + std::to_string(int(1 / totalTime)), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
        
      return image;
    }
};

DnnParams myNet;

class OpencvNode : public rclcpp::Node
{
  public:
    OpencvNode()
    : Node("opencv_node")
    {
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10, std::bind(&OpencvNode::imageCallback, this, _1));

      image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_topic", 10);
    }

  private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      cv_bridge::CvImagePtr cv_ptr;
      cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
      cv::Mat cvImage = cv_ptr->image;
      if (cvImage.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load image");
            return;
        }
      cv::resize(cvImage, cvImage, {640, 480});
      cvImage = myNet.process(cvImage);

      sensor_msgs::msg::Image::SharedPtr rosImage = cv_bridge::CvImage(msg->header, "bgr8", cvImage).toImageMsg();
      image_pub_->publish(*rosImage);
    }
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpencvNode>());
  rclcpp::shutdown();
  return 0;
}