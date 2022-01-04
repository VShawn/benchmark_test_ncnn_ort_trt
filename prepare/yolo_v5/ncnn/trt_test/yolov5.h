#pragma once

#include <string>
#include <vector>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>  // nvidia加载模型进行推理的插件
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifndef _DetectionBox_
#define _DetectionBox_
typedef struct
{
	unsigned int label; // for general object detection.
	int left;
	int top;
	int right;
	int bottom;
	double score;
} DetectionBox;
#endif

// 自定义配置结构
struct Configuration {
	float confThreshold;  // Confidence threshold
	float nmsThreshold;   // Non-maximum suppression threshold
	float objThreshold;   // Object Confidence threshold
	std::string modelpath;
	std::string detectorName;
	int width;
	int height;
};

class YOLOv5 {
public:
	//YOLOv5(Configuration config);
	~YOLOv5();
	void UnInit();
	void Init(Configuration configuration);
	/// <summary>
	/// 执行推理
	/// </summary>
	/// <param name="frame">图片</param>
	/// <param name="obj_result">返回bbox</param>
	/// <param name="vec_class_in">筛选bbox类型，为空时表示返回所有类型的 bbox</param>
	/// <param name="vec_class_not_in">剔除bbox类型，vecClassTypeIn为空时起效，表示将其中的类型排除出去，为空时啥也不排除</param>
	void detect(
		cv::Mat& frame,
		std::vector<DetectionBox>& obj_results,
		const std::vector<short>& vec_class_in = {},
		const std::vector<short>& vec_class_not_in = {});
	void detect_only();
	std::string DetectorName;
	static bool build_model(std::string onnx_path, std::string build_path, int w);
private:
	void nms(std::vector<DetectionBox>& input_boxes);
	cv::Mat resize_image(cv::Mat srcimg, int* newh, int* neww, int* top,
		int* left);

	void loadTrt(const std::string strName);


private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
	std::vector<std::string> class_names;  // 类别名称
	const bool keep_ratio = true;
	nvinfer1::ICudaEngine* m_CudaEngine;
	nvinfer1::IRuntime* m_CudaRuntime;
	nvinfer1::IExecutionContext* m_CudaContext;
	cudaStream_t m_CudaStream;            //初始化流,CUDA流的类型为cudaStream_t
	int m_iInputIndex;
	int m_iOutputIndex;
	int m_iClassNums;
	int m_iBoxNums;
	cv::Size m_InputSize;
	void* m_ArrayDevMemory[2]{ 0 };
	void* m_ArrayHostMemory[2]{ 0 };
	int m_ArraySize[2]{ 0 };
	std::vector<cv::Mat> m_InputWrappers{};   // 空的matvector
	bool hasInited = false;
};

