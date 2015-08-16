#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>

#include <unistd.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;

int m, rows, cols;

ifstream in_image;
ifstream in_label;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int main(int argc, char** argv)
{
	char *image_filename = argv[1], *label_filename = argv[2];

	char *output_folder = argv[3];

	in_image.open(image_filename, ios::binary);
	in_label.open(label_filename, ios::binary);

	// read the imagine numbers
	int imagine_number;
	in_image.read((char*)&imagine_number, sizeof(imagine_number));
	in_label.read((char*)&imagine_number, sizeof(imagine_number));

	in_image.read((char*)&m, sizeof(m));
	in_label.read((char*)&m, sizeof(m));
	m = reverseInt(m);
	cout<<"Number of Pictures = "<<m<<endl;

	in_image.read((char*)&rows, sizeof(rows));
	rows = reverseInt(rows);
	cout<<"Number of Rows = "<<rows<<endl;
	in_image.read((char*)&cols, sizeof(cols));
	cols = reverseInt(cols);
	cout<<"Number of Columns = "<<cols<<endl;

	unsigned char tmp;
	unsigned char label;
	cv::Mat image(28, 28, CV_32FC1);

	for(int i=0; i<1000; ++i) 
	{
		cout<<"Processing "<<i<<"th picture!"<<endl;
	
		for(int r=0; r<rows; ++r)
			for(int c=0; c<cols; ++c) 
			{
				in_image.read((char*)&tmp, sizeof(tmp));
				image.at<float>(r, c) = tmp;
			}

		in_label.read((char*)&label, sizeof(label));
		
		char file_name[20];
		sprintf(file_name, "%d_%d.jpg", i, label);
		
		cout<<"\t Storing to file "<<file_name<<endl;

		chdir(output_folder);
		cv::imwrite(file_name, image);
		chdir("..");
	}
}
