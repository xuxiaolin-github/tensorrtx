#include "circuqueue.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

int yolov5s_inference(CircuQueue<cv::Mat> * );


int yolov5s_serialize(std::string wts_name, std::string engine_name,std::string net,float d = 0.0f , float w = 0.0f );




