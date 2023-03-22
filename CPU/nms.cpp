#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <list>
#include <map>
using namespace cv;
using namespace std;

struct Detect2dObject {
  float x;
  float y;
  float w;
  float h;
  float class_prob;
  int class_idx;
};

template <typename T>
static inline bool compObject(const T &lhs, const T &rhs) {
  return lhs.class_prob > rhs.class_prob;
}
float iou(const cv::Rect &a, const cv::Rect &b) {
  float inter_area = (a & b).area();
  float union_area = a.area() + b.area() - inter_area;

  return inter_area / union_area;
}
float distance(const cv::Rect &a, const cv::Rect &b){
	cv::Point center1 = {a.x + a.width /2, a.y + a.height /2};
	cv::Point center2 = {b.x + b.width /2, b.y + b.height /2};
	return sqrt(pow(center1.x-center2.x,2) + pow(center1.y-center2.y,2));
}
// 1，提前排好序 2，已经为false的不需要再次进行iou比较 3，对于框的距离比较远的直接排除
void nms2d(std::map<int, std::list<Detect2dObject>> &m, float nms_thresh) {
	constexpr int DISTANCE = 100;
	for (auto &e : m) {
		auto &l = e.second;
		l.sort(compObject<Detect2dObject>);

		// compute iou
		for (auto it = l.begin(); it != l.end(); it++) {
		cv::Rect a(it->x, it->y, it->w, it->h);
		auto cursor = it;
		std::advance(cursor, 1);

		for (; cursor != l.end();) {
			cv::Rect b(cursor->x, cursor->y, cursor->w, cursor->h);
			if(distance(a, b) > DISTANCE) {
				cursor++;
			} else {
				if (iou(a, b) > nms_thresh) {
					cursor = l.erase(cursor);
				} else {
					cursor++;
				}
			}
		}
		}
	}
}

int main()
{
	std::ifstream in,in1;
    std::string line; 
  	
	int count=6000;
	std::map<int,std::list<Detect2dObject>> m;
	in.open("../boxes.txt"); //y1, x1, y2, x2
	in1.open("../scores.txt");
    if (in.is_open() && in1.is_open()) {
        int class_id = 0;
        while(getline(in, line)){
			istringstream iss(line);
            Detect2dObject tmp;
			iss >> tmp.y;
			iss >> tmp.x;
			iss >> tmp.h;
			iss >> tmp.w;
			tmp.h -= tmp.y; //y2 -> h
            tmp.w -= tmp.x; //x2 -> w
			in1 >> tmp.class_prob;
			tmp.class_idx = 0;
			tmp.y *= 640;
			tmp.x *= 640;
			tmp.h *= 640;
			tmp.w *= 640;

			if (m.find(0) == m.end()) {
				m[class_id] = std::list<Detect2dObject>{std::move(tmp)};
			} else {
				m[class_id].emplace_back(std::move(tmp));
			}
        }
    }
    in.close();
	in1.close();
	
	// for (auto &e : m) {
	// 	auto &l = e.second;
	// 	for (auto it = l.begin(); it != l.end(); it++) {
	// 		cout << it->x << " "<< it->y << " "<< it->h << " "<< it->w << " " << it->class_prob << endl; 
	// 		getchar();
	// 	}
	// }

	float nms_thresh = 0.6;
	nms2d(m,nms_thresh);
	
	for (auto &e : m) {
		cout << " result_num " << e.second.size() << endl; 
	}
	return 0;
}
