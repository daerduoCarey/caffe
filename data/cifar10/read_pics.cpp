#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>

#include <unistd.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;

int m, rows, cols, rgb;

char* names[10] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

ifstream in;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int main(int argc, char** argv)
{
	char *image_filename = argv[1];
	char *output_folder = argv[2];

	in.open(image_filename, ios::binary);

	m = 10000;
	rows = 32; cols = 32; rgb = 3;

	unsigned char tmp;
	unsigned char label;

	cv::Mat image(rows, cols, CV_8UC3);
	float data[rgb][rows][cols];

	for(int i=0; i<1000; ++i) 
	{
		cout<<"Processing "<<i<<"th picture!"<<endl;
	
		in.read((char*)&label, sizeof(label));
		
		for(int color=0; color<rgb; ++color)
			for(int r=0; r<rows; ++r)
				for(int c=0; c<cols; ++c) 
				{
					in.read((char*)&tmp, sizeof(tmp));
					data[color][r][c] = tmp;
				}

		for(int r=0; r<rows; ++r)
			for(int c=0; c<cols; ++c)
			{
				image.at<cv::Vec3b>(r, c) = cv::Vec3b(data[0][r][c], data[1][r][c], data[2][r][c]);
			}

		char file_name[20];
		sprintf(file_name, "%d_%s.jpg", i, names[label]);
		
		cout<<"\t Storing to file "<<file_name<<endl;

		chdir(output_folder);
		cv::imwrite(file_name, image);
		chdir("..");
	}
}
