#include "yolov5.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <sys/stat.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include "loghelper.h"

// �����ռ�
using namespace cv;
using namespace nvinfer1;

static void my_assert(bool exp, int line = 0) {
	if (!exp)
	{
		LOG_FATAL("my_assert failure: %d", line);
		throw "my_assert failure";
	}
}

// Logger for TRT info/warning/errors, https://github.com/onnx/onnx-tensorrt/blob/main/onnx_trt_backend.cpp
class TRT_Logger : public nvinfer1::ILogger {
	nvinfer1::ILogger::Severity _verbosity;
	std::ostream* _ostream;

public:
	TRT_Logger(Severity verbosity = Severity::kWARNING,
		std::ostream& ostream = std::cout) : _verbosity(verbosity), _ostream(&ostream) {}
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= _verbosity) {

			//const char* sevstr =
			//	(severity == Severity::kINTERNAL_ERROR
			//		? "    BUG"
			//		: severity == Severity::kERROR
			//		? "  ERROR"
			//		: severity == Severity::kWARNING
			//		? "WARNING"
			//		: severity == Severity::kINFO ? "   INFO"
			//		: "UNKNOWN");
			//time_t rawtime = std::time(0);
			//char buf[256];
			//strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
			//(*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;

			std::stringstream ss;
			ss << "TensorRT: " << msg;
			auto x = (severity == Severity::kINTERNAL_ERROR
				? log_helper::enum_level::fatal
				: severity == Severity::kERROR
				? log_helper::enum_level::error
				: severity == Severity::kWARNING
				? log_helper::enum_level::warning
				: severity == Severity::kINFO ? log_helper::enum_level::info
				: log_helper::enum_level::debug);
			log_helper::log.output(x, ss.str().c_str());
		}
	}
};

// �ж��ļ��Ƿ��γ�
static bool ifFileExists(const char* FileName) {
	struct stat my_stat;
	return (stat(FileName, &my_stat) == 0);
}


/**
*���� nvinfer1 �����ռ䣬�Է��� TensorRT 8.5 �汾�� API��
�ڴ������ļ���ʱ����Ҫָ�� std::ios::binary �������Ա�֤�������ļ��Ķ�ȡ��д�뷽ʽ��
�����л��ļ��Ķ�ȡ�����ϣ�ʹ����һ�� std::stringstream ������ std::ifstream::rdbuf() ���� std::istreambuf_iterator<char>(fin) ��ȫ�����룬�Ա��ڹ۲����л����ݵ����͡�
�ڷ����л�����ʱ�����������л����ݴ洢�ڻ������У���ʹ�ô�С��ȫ�Ļ�����ָ��ʹ�С���ݸ�������
�ڴ��� TensorRT ִ��������ʱ�����ж������Ƿ�Ϊ�գ�������ֳ�����������⡣
��Ҫע����ǣ����޸ĺ��Ż�����ʱ��Ӧ�ý��г�ֲ��Ժ���֤��ȷ���������ȷ�Ժͽ�׳�ԡ�
ͬʱ����Ҫע�� TensorRT �汾�ļ����ԣ�ȷ���ڲ�ͬ�汾�� TensorRT �϶��ܹ���ȷ���д��롣
*/
void YOLOv5::loadTrt(const std::string strName) {
	TRT_Logger gLogger;
	// ���� TensorRT ����ʱ����
	m_CudaRuntime = nvinfer1::createInferRuntime(gLogger);
	// �������ļ�����ȡ���л�����
	std::ifstream fin(strName, std::ios::binary);
	if (!fin.good()) {
		fin.close();
		throw std::runtime_error("Unable to open engine file: " + strName);
	}
	// ��ȡ�ļ���С
	fin.seekg(0, std::ios::end);
	size_t fileSize = fin.tellg();
	fin.seekg(0, std::ios::beg);
	// ���仺�������洢���л�����
	std::unique_ptr<char[]> engineBuffer(new char[fileSize]);
	fin.read(engineBuffer.get(), fileSize);
	fin.close();
	// �����л�����
	m_CudaEngine = m_CudaRuntime->deserializeCudaEngine(engineBuffer.get(),
		fileSize, nullptr);
	if (!m_CudaEngine) {
		throw std::runtime_error("Failed to deserialize engine from file: " +
			strName);
	}
	// ���� TensorRT ִ��������
	m_CudaContext = m_CudaEngine->createExecutionContext();
	// �ͷ��ڴ�
	m_CudaRuntime->destroy();
}

// ��ʼ��

void YOLOv5::Init(Configuration configuration) {
	confThreshold = configuration.confThreshold;
	nmsThreshold = configuration.nmsThreshold;
	objThreshold = configuration.objThreshold;
	inpHeight = configuration.height;
	inpWidth = configuration.width;
	DetectorName = configuration.detectorName;

	LOG_DEBUG("YOLO(%s) init with conf_thre %f, iou_thre %f, objThreshold %f", DetectorName, confThreshold, nmsThreshold, objThreshold);

	std::string model_path = configuration.modelpath;  // ģ��Ȩ��·��
	// ����ģ��
	std::string strTrtName = configuration.modelpath;  // ����ģ��Ȩ��
	size_t sep_pos = model_path.find_last_of(".");
	strTrtName = model_path.substr(0, sep_pos) + ".engine";
	if (ifFileExists(strTrtName.c_str())) {
		loadTrt(strTrtName);
	}
	else {
		LOG_ERROR("TR(%s) file do not existed!", strTrtName);
		exit(-1);
	}

	// ���ü��ص�ģ�ͻ�ȡ���������Ϣ
	// ʹ����������blob������ȡ������������
	m_iInputIndex = m_CudaEngine->getBindingIndex("images");    // ��������
	m_iOutputIndex = m_CudaEngine->getBindingIndex("output0");  // ���
	Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInputIndex);  // ���룬ά��[0,1,2,3]NHWC
	Dims dims_o = m_CudaEngine->getBindingDimensions(m_iOutputIndex);  // ���

	dims_i.d[0] = 1;
	dims_o.d[0] = 1;
//#ifdef _DEBUG

	std::cout << std::endl << std::endl << "====================== model info(" << strTrtName << ") =========================" << std::endl;
	// ���������������
	std::cout << "type = ";
	switch (m_CudaEngine->getBindingDataType(m_iInputIndex)) {
	case nvinfer1::DataType::kFLOAT:
		std::cout << "FP32";
		break;
	case nvinfer1::DataType::kHALF:
		std::cout << "FP16";
		break;
	case nvinfer1::DataType::kINT8:
		std::cout << "INT8";
		break;
	case nvinfer1::DataType::kINT32:
		std::cout << "INT32";
		break;
	default:
		std::cout << "unknown";
		break;
	}
	std::cout << std::endl;
	std::cout << "input dims�� " << dims_i.nbDims << " " << dims_i.d[0] << " "
		<< dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3] << std::endl;
	std::cout << "output dims�� " << dims_o.nbDims << " " << dims_o.d[0] << " "
		<< dims_o.d[1] << " " << dims_o.d[2] << std::endl;
	std::cout << "==============================================================================" << std::endl << std::endl << std::endl;
//#endif // _DEBUG

	int size1 = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];  // չƽ
	int size2 = dims_o.d[0] * dims_o.d[1] * dims_o.d[2];  // ���д�С

	m_InputSize = cv::Size(dims_i.d[3], dims_i.d[2]);  // ����ߴ�(W,H)
	m_iClassNums = dims_o.d[2] - 5;                    // �������
	m_iBoxNums = dims_o.d[1];                          // num_pre_boxes

	// �����ڴ��С
	my_assert(m_iInputIndex < 2 && m_iInputIndex >= 0, __LINE__);
	my_assert(m_iOutputIndex < 2 && m_iOutputIndex >= 0, __LINE__);
	cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], size1 * sizeof(float));  // ��CUDA�Դ��Ϸ���size1 * sizeof(float)��С���ڴ�ռ䣬�������Դ�ָ�븳ֵ��m_ArrayDevMemory[m_iInputIndex]
	cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], size2 * sizeof(float));  // ͬ��
	m_ArrayHostMemory[m_iInputIndex] = malloc(size1 * sizeof(float));  // ���������ڴ��Ϸ���һ���СΪsize1 * sizeof(float)���ڴ�ռ䣬�����õ�ַ��ֵ��m_ArrayHostMemory[m_iInputIndex]
	m_ArrayHostMemory[m_iOutputIndex] = malloc(size2 * sizeof(float));  //
	m_ArraySize[m_iInputIndex] = size1 * sizeof(float);  // ����ǰ�������ݵ��ڴ�ռ��С��¼��m_ArraySize�����еĶ�Ӧλ�ã��Է���������ڴ洫�������
	m_ArraySize[m_iOutputIndex] = size2 * sizeof(float);  //

	my_assert(m_ArrayDevMemory[m_iInputIndex] != NULL && m_ArrayDevMemory[m_iInputIndex] != nullptr, __LINE__);
	my_assert(m_ArrayDevMemory[m_iOutputIndex] != NULL && m_ArrayDevMemory[m_iOutputIndex] != nullptr, __LINE__);
	my_assert(m_ArrayHostMemory[m_iInputIndex] != NULL && m_ArrayHostMemory[m_iInputIndex] != nullptr, __LINE__);
	my_assert(m_ArrayHostMemory[m_iOutputIndex] != NULL && m_ArrayHostMemory[m_iOutputIndex] != nullptr, __LINE__);
	my_assert(m_ArraySize[m_iInputIndex] >= 0, __LINE__);
	my_assert(m_ArraySize[m_iOutputIndex] >= 0, __LINE__);

	// ֪ʶ�㣺
	// 1. emplace_back()��������ֱ����vectorβ������Ԫ�أ������Ǵ�����ʱ����Ȼ���俽����vector�С��������Ա�����ÿ������캯�����ƶ����캯�������Ч�ʡ�
	// 2. ����Ĺ��죺cv::Mat(int rows, int cols, int type, void* data, size_t step = AUTO_STEP);

	m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);  // ��m_ArrayHostMemory[m_iInputIndex]ָ���һ�οռ䰴��ÿ��Ԫ�ش�СΪһ��float�ķ�ʽ������һ������Mat������ά��Ϊdims_i.d[2]��dims_i.d[3]������������CV_32FC1��32λ��������ͨ������
	m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, (char*)m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3]);  // visual studio ��֧�ֶ�void*��ָ��ֱ�����㣬��Ҫת���ɾ������͡�
	m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, (char*)m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);

	// ��ȡ�����Ϣ
	/*std::string classesFile =
		"./class.names";
	std::ifstream ifs(classesFile.c_str());
	std::string line;
	while (getline(ifs, line)) this->class_names.push_back(line);*/
	LOG_DEBUG("Finish Init TensorRT");

	hasInited = true;
}

void YOLOv5::UnInit()
{
	LOG_INFO("YOLO(%s) UnInit", DetectorName, confThreshold, nmsThreshold, objThreshold);
	for (auto& p : m_ArrayDevMemory) {
		cudaFree(p);
		p = nullptr;
	}
	for (auto& p : m_ArrayHostMemory) {
		free(p);
		p = nullptr;
	}
	cudaStreamDestroy(m_CudaStream);
	//m_CudaContext->destroy();  // �����ͷŸ�CUDA������ʹ�õ�CUDA��Դ
	//m_CudaEngine->destroy();   // ���ͷŸ�TensorRTģ�͵�CUDA�������
	hasInited = false;
}

YOLOv5::~YOLOv5() { UnInit(); }

Mat YOLOv5::resize_image(Mat srcimg, int* newh, int* neww, int* top,
	int* left) {
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left,
				this->inpWidth - *neww - *left, BORDER_CONSTANT,
				cv::Scalar(114, 114, 114));
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0,
				BORDER_CONSTANT, cv::Scalar(114, 114, 114));
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLOv5::nms(std::vector<DetectionBox>& input_boxes) {
	sort(input_boxes.begin(), input_boxes.end(), [](DetectionBox a, DetectionBox b) { return a.score > b.score; });  // ��������
	std::vector<bool> remove_flags(input_boxes.size(), false);
	auto iou = [](const DetectionBox& box1, const DetectionBox& box2) {
		float xx1 = max(box1.left, box2.left);
		float yy1 = max(box1.top, box2.top);
		float xx2 = min(box1.right, box2.right);
		float yy2 = min(box1.bottom, box2.bottom);
		// ����
		float w = max(0.0f, xx2 - xx1 + 1);
		float h = max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// ����
		float union_area =
			max(0, box1.right - box1.left) * max(0, box1.bottom - box1.top) +
			max(0, box2.right - box2.left) * max(0, box2.bottom - box2.top) -
			inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i) {
		if (remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j) {
			if (remove_flags[j]) continue;
			if (input_boxes[i].label == input_boxes[j].label &&
				iou(input_boxes[i], input_boxes[j]) >= this->nmsThreshold) {
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
	// remove_if()���� remove_if(beg, end, op) // �Ƴ�����[beg,end)��ÿһ�������ж�ʽ:op(elem)���true����Ԫ��
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(),
		[&idx_t, &remove_flags](const DetectionBox& f) {
			return remove_flags[idx_t++];
		}),
		input_boxes.end());
}

void YOLOv5::detect(cv::Mat& frame, std::vector<DetectionBox>& obj_results, const std::vector<short>& vec_class_in, const std::vector<short>& vec_class_not_in)
{
	if (hasInited == false) {
		LOG_FATAL("Use detector befor initialization", DetectorName);
	}

	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);

	cv::cvtColor(dstimg, dstimg, cv::COLOR_BGR2RGB);  // ��BGRת��RGB
	cv::Mat m_Normalized;
	dstimg.convertTo(m_Normalized, CV_32FC3, 1 / 255.);
	// ��m_Normalized��ÿ��ͨ�����Ϊһ������Ȼ����Щ����˳��洢��m_InputWrappers�������У����滻�������ݡ�
	cv::split(m_Normalized, m_InputWrappers);  // ͨ������[h,w,3] RGB

	// ����CUDA��,����ʱTensorRTִ��ͨ�����첽�ģ���˽��ں�����CUDA��
	cudaStreamCreate(&m_CudaStream);
	auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream);
	auto start = std::chrono::system_clock::now();
	auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);  // TensorRT ִ��ͨ�����첽�ģ���˽��ں����� CUDA ����
	ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream);  //������ظ�CPU�����ݴ��Դ浽�ڴ�
	ret = cudaStreamSynchronize(m_CudaStream);
}

void YOLOv5::detect_only()
{
	// ����CUDA��,����ʱTensorRTִ��ͨ�����첽�ģ���˽��ں�����CUDA��
	cudaStreamCreate(&m_CudaStream);
	auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream);
	auto start = std::chrono::system_clock::now();
	auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);  // TensorRT ִ��ͨ�����첽�ģ���˽��ں����� CUDA ����
	ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream);  //������ظ�CPU�����ݴ��Դ浽�ڴ�
	ret = cudaStreamSynchronize(m_CudaStream);
}
//void main(void)
//{
//    Configuration configuration;
//    configuration.confThreshold = 0.1;
//    configuration.modelpath = "yolov5_1280_input.engine";
//    configuration.nmsThreshold = 0.45;
//    configuration.objThreshold = 0.1;
//
//    cv::Mat img = cv::imread("1-1.jpg");
//    YOLOv5 yolov5(configuration);
//    yolov5.detect(img);
//}




bool YOLOv5::build_model(std::string onnx_path, std::string build_path, int w) {
	TRT_Logger logger;

	// �����builder, config, network�ǻ�����Ҫ�����
	// ��������������Ҫһ��builderȥbuild������磬���������нṹ������ṹ�����в�ͬ������
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	// ����һ���������ã�ָ��TensorRTӦ������Ż�ģ�ͣ�tensorRT���ɵ�ģ��ֻ�����ض�����������
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// �������綨�壬����createNetworkV2(1)��ʾ��������batch size���°�tensorRT(>=7.0)ʱ�����������0������batch size
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	// onnx parser������������onnxģ��
	auto parser = nvonnxparser::createParser(*network, logger);
	if (!parser->parseFromFile(onnx_path.c_str(), 1)) {
		printf("Failed to parse classifier.onnx.\n");
		return false;
	}

	// ���ù�������С
	printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);

	// ��Ҫͨ��profile��ʹ��batchsizeʱ��̬�ɱ�ģ���������֮ǰ����onnxָ���Ķ�̬batchsize�Ƕ�Ӧ��
	int maxBatchSize = 10;
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();

	// ����batchsize�����/��С/����ֵ
	input_dims.d[0] = 1;
	input_dims.d[2] = w;
	input_dims.d[3] = w;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);

	// ��ʼ����tensorrtģ��engine
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	if (engine == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	// �������õ�tensorrtģ��engine�����л���������ļ���
	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen(build_path.c_str(), "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	// ����destory��ָ��
	model_data->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();

	printf("Build Done.\n");
	return true;
}