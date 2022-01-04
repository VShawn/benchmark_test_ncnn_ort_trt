// trt_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "yolov5.h"

#include <filesystem>

inline bool ends_with(std::string const& value, std::string const& ending)
{
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
inline bool starts_with(std::string const& value, std::string const& ending)
{
	if (ending.size() > value.size()) return false;
	return std::equal(ending.begin(), ending.end(), value.begin());
}

void Test(const std::string& _onnx_path, cv::Mat img, const int input_width, const int input_height,
	const int loop = 100)
{
	std::string trt_path = _onnx_path;
	if (ends_with(_onnx_path, ".engine") == false)
	{
		if (ends_with(_onnx_path, ".onnx"))
		{
			trt_path = _onnx_path + ".engine";
			if (!std::filesystem::exists(trt_path))
			{
				if (!YOLOv5::build_model(_onnx_path, trt_path, input_width)) {
					std::cout << "engine 生成失败！" << std::endl;
					exit(-1);
				}
			}
		}
		else
			trt_path = trt_path + ".engine";
	}

	YOLOv5 detector;
	Configuration configuration;
	configuration.confThreshold = 0.3;
	configuration.modelpath = trt_path;
	configuration.nmsThreshold = 0.35;
	configuration.objThreshold = 0.3;
	configuration.height = input_width;
	configuration.width = input_height;
	configuration.detectorName = _onnx_path;
	detector.Init(configuration);
	std::vector<DetectionBox> bbox_results;
	detector.detect(img, bbox_results);

	clock_t start = clock();
	for (int i = 0; i < loop; ++i)
	{
		detector.detect_only();
	}
	clock_t end = clock();
	double time2 = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "trt " << _onnx_path << " " << input_height << " time avg - cost " << time2 / loop << " s, " << loop / time2 << " fps" << std::endl;//输出运行时间
	while (1)
	{
		std::cout << "当前加载 1 个模型，输入 c 加载第2模型" << std::endl;
		if (std::getchar() == 'c')
			break;
	}

	{
		YOLOv5 detector;
		Configuration configuration;
		configuration.confThreshold = 0.3;
		configuration.modelpath = _onnx_path;
		configuration.nmsThreshold = 0.35;
		configuration.objThreshold = 0.3;
		configuration.height = input_width;
		configuration.width = input_height;
		configuration.detectorName = _onnx_path;
		detector.Init(configuration);
		std::vector<DetectionBox> bbox_results;
		detector.detect(img, bbox_results);

		while (1)
		{
			std::cout << "当前加载 2 个模型，且已完成预热，请查看资源消耗, 输入 c 继续后续测试" << std::endl;
			if (std::getchar() == 'c')
				break;
		}
	}
}



int main()
{
	cv::Mat img = cv::imread("C:\\dog.310.jpg");
	if (img.rows == 0)
	{
		std::cout << "请将 dog.310.jpg 放置在 C 盘根目录下" << std::endl;
		return 1;
	}
	while (true)
	{
		std::cout << "输入 e 退出，输入 n|n6|s|s6|...|l|l6 以运行对应的 yolo 模型" << std::endl;
		std::string c;
		std::cin >> c;
		int size = 640;
		// c 以 6 结尾，表示输入尺寸为 1280
		if (c[c.size() - 1] == '6')
		{
			size = 1280;
		}
		std::string model_name = "yolov5" + c;
		Test(model_name, img, size, size);
	}
}