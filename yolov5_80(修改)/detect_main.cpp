#include "yolov5.h"
#include "circuqueue.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

int main(int argc, char *argv[]){

    
	
	// yolov5s_serialize("../yolov5s.wts","../yolov5s.engine","s");
	
	CircuQueue<cv::Mat> Queue(200);
	
	cv::VideoCapture capture("/media/xu/HSAX/green_park_videos/videos/2021-11-16/2/2021-11-16-115708.mp4");

	std::thread thread_stream = std::thread(yolov5s_inference,&Queue);
	
	
	thread_stream.detach();
	

	int frame_num=1;
	
	
	// double rate;
	while(1){

		cv::Mat frame;
		
		capture>>frame;
		

		if (frame.empty())
		{
			// rate=capture.get(cv::CAP_PROP_FPS);
			capture.set(cv::CAP_PROP_POS_FRAMES, 0);
			capture>>frame;
			// break;
		}
		// cv::resize(frame,frame, cv::Size(640, 640));
        // cvtColor(frame,frame,cv::COLOR_BGR2RGBA);
		
		Queue.enqueue(frame);
		
		// cv::imshow("读取视频",frame);
		// cv::waitKey(1);
		// printf("%d-",frame_num++);
		
		
	}
	
	
		

	return 0;

}
