#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <unistd.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <iostream>

IplImage* image = 0; 
IplImage* prev_image = 0;
CvCapture* capture = 0;
std::string out_file;
char temp [30];
int show = 0; 


int main( int argc, char** argv )
{
	int frameNum = 0;

	char* video = argv[1];
	capture = cvCreateFileCapture(video);

	if( !capture ) { 
		printf( "Could not initialize capturing..\n" );
		return -1;
	}
	
	if( show == 1 )
		cvNamedWindow( "Video", 0 );

	while( true ) {
		IplImage* frame = 0;
		int c;
		// get a new frame
		frame = cvQueryFrame( capture );
		if( !frame )
			break;

		if( !image ) {
			image =  cvCreateImage( cvSize(frame->width,frame->height), 8, 3 );
			image->origin = frame->origin;
		}

		cvCopy( frame, image, 0 );

		if( show == 1 ) {
			cvShowImage( "Video", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
		std::ostringstream os;
		os<< "saved/out_"<<frameNum<<".jpg";
		std::string out_file=os.str();
		std::cerr<<out_file<<std::endl;
		cvSaveImage(strcpy(temp,out_file.c_str()),frame);
		std::cerr << "The " << frameNum << "-th frame" << std::endl;
		frameNum++;
	}

	if( show == 1 )
		cvDestroyWindow("Video");

	return 0;
}
