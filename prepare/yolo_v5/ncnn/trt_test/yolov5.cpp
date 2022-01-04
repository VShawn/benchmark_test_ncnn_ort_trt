#include "yolov5.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <sys/stat.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include "loghelper.h"

// 命名空间
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

// 判断文件是否形成
static bool ifFileExists(const char* FileName) {
	struct stat my_stat;
	return (stat(FileName, &my_stat) == 0);
}


/**
*引入 nvinfer1 命名空间，以访问 TensorRT 8.5 版本的 API。
在打开引擎文件的时候需要指定 std::ios::binary 参数，以保证二进制文件的读取和写入方式。
在序列化文件的读取方法上，使用了一个 std::stringstream 来承载 std::ifstream::rdbuf() 或者 std::istreambuf_iterator<char>(fin) 的全部读入，以便于观察序列化数据的类型。
在反序列化引擎时，将引擎序列化数据存储在缓冲区中，并使用大小安全的缓冲区指针和大小传递给函数。
在创建 TensorRT 执行上下文时，先判断引擎是否为空，避免出现程序崩溃等问题。
需要注意的是，在修改和优化代码时，应该进行充分测试和验证，确保代码的正确性和健壮性。
同时，需要注意 TensorRT 版本的兼容性，确保在不同版本的 TensorRT 上都能够正确运行代码。
*/
void YOLOv5::loadTrt(const std::string strName) {
	TRT_Logger gLogger;
	// 创建 TensorRT 运行时对象
	m_CudaRuntime = nvinfer1::createInferRuntime(gLogger);
	// 打开引擎文件，读取序列化数据
	std::ifstream fin(strName, std::ios::binary);
	if (!fin.good()) {
		fin.close();
		throw std::runtime_error("Unable to open engine file: " + strName);
	}
	// 获取文件大小
	fin.seekg(0, std::ios::end);
	size_t fileSize = fin.tellg();
	fin.seekg(0, std::ios::beg);
	// 分配缓冲区，存储序列化数据
	std::unique_ptr<char[]> engineBuffer(new char[fileSize]);
	fin.read(engineBuffer.get(), fileSize);
	fin.close();
	// 反序列化引擎
	m_CudaEngine = m_CudaRuntime->deserializeCudaEngine(engineBuffer.get(),
		fileSize, nullptr);
	if (!m_CudaEngine) {
		throw std::runtime_error("Failed to deserialize engine from file: " +
			strName);
	}
	// 创建 TensorRT 执行上下文
	m_CudaContext = m_CudaEngine->createExecutionContext();
	// 释放内存
	m_CudaRuntime->destroy();
}

// 初始化

void YOLOv5::Init(Configuration configuration) {
	confThreshold = configuration.confThreshold;
	nmsThreshold = configuration.nmsThreshold;
	objThreshold = configuration.objThreshold;
	inpHeight = configuration.height;
	inpWidth = configuration.width;
	DetectorName = configuration.detectorName;

	LOG_DEBUG("YOLO(%s) init with conf_thre %f, iou_thre %f, objThreshold %f", DetectorName, confThreshold, nmsThreshold, objThreshold);

	std::string model_path = configuration.modelpath;  // 模型权重路径
	// 加载模型
	std::string strTrtName = configuration.modelpath;  // 加载模型权重
	size_t sep_pos = model_path.find_last_of(".");
	strTrtName = model_path.substr(0, sep_pos) + ".engine";
	if (ifFileExists(strTrtName.c_str())) {
		loadTrt(strTrtName);
	}
	else {
		LOG_ERROR("TR(%s) file do not existed!", strTrtName);
		exit(-1);
	}

	// 利用加载的模型获取输入输出信息
	// 使用输入和输出blob名来获取输入和输出索引
	m_iInputIndex = m_CudaEngine->getBindingIndex("images");    // 输入索引
	m_iOutputIndex = m_CudaEngine->getBindingIndex("output0");  // 输出
	Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInputIndex);  // 输入，维度[0,1,2,3]NHWC
	Dims dims_o = m_CudaEngine->getBindingDimensions(m_iOutputIndex);  // 输出

	dims_i.d[0] = 1;
	dims_o.d[0] = 1;
//#ifdef _DEBUG

	std::cout << std::endl << std::endl << "====================== model info(" << strTrtName << ") =========================" << std::endl;
	// 输出处理数据类型
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
	std::cout << "input dims： " << dims_i.nbDims << " " << dims_i.d[0] << " "
		<< dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3] << std::endl;
	std::cout << "output dims： " << dims_o.nbDims << " " << dims_o.d[0] << " "
		<< dims_o.d[1] << " " << dims_o.d[2] << std::endl;
	std::cout << "==============================================================================" << std::endl << std::endl << std::endl;
//#endif // _DEBUG

	int size1 = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];  // 展平
	int size2 = dims_o.d[0] * dims_o.d[1] * dims_o.d[2];  // 所有大小

	m_InputSize = cv::Size(dims_i.d[3], dims_i.d[2]);  // 输入尺寸(W,H)
	m_iClassNums = dims_o.d[2] - 5;                    // 类别数量
	m_iBoxNums = dims_o.d[1];                          // num_pre_boxes

	// 分配内存大小
	my_assert(m_iInputIndex < 2 && m_iInputIndex >= 0, __LINE__);
	my_assert(m_iOutputIndex < 2 && m_iOutputIndex >= 0, __LINE__);
	cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], size1 * sizeof(float));  // 在CUDA显存上分配size1 * sizeof(float)大小的内存空间，并将该显存指针赋值给m_ArrayDevMemory[m_iInputIndex]
	cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], size2 * sizeof(float));  // 同上
	m_ArrayHostMemory[m_iInputIndex] = malloc(size1 * sizeof(float));  // 于在主机内存上分配一块大小为size1 * sizeof(float)的内存空间，并将该地址赋值给m_ArrayHostMemory[m_iInputIndex]
	m_ArrayHostMemory[m_iOutputIndex] = malloc(size2 * sizeof(float));  //
	m_ArraySize[m_iInputIndex] = size1 * sizeof(float);  // 将当前输入数据的内存空间大小记录在m_ArraySize数组中的对应位置，以方便后续的内存传输操作。
	m_ArraySize[m_iOutputIndex] = size2 * sizeof(float);  //

	my_assert(m_ArrayDevMemory[m_iInputIndex] != NULL && m_ArrayDevMemory[m_iInputIndex] != nullptr, __LINE__);
	my_assert(m_ArrayDevMemory[m_iOutputIndex] != NULL && m_ArrayDevMemory[m_iOutputIndex] != nullptr, __LINE__);
	my_assert(m_ArrayHostMemory[m_iInputIndex] != NULL && m_ArrayHostMemory[m_iInputIndex] != nullptr, __LINE__);
	my_assert(m_ArrayHostMemory[m_iOutputIndex] != NULL && m_ArrayHostMemory[m_iOutputIndex] != nullptr, __LINE__);
	my_assert(m_ArraySize[m_iInputIndex] >= 0, __LINE__);
	my_assert(m_ArraySize[m_iOutputIndex] >= 0, __LINE__);

	// 知识点：
	// 1. emplace_back()方法可以直接在vector尾部构造元素，而不是创建临时对象，然后将其拷贝到vector中。这样可以避免调用拷贝构造函数和移动构造函数，提高效率。
	// 2. 矩阵的构造：cv::Mat(int rows, int cols, int type, void* data, size_t step = AUTO_STEP);

	m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);  // 将m_ArrayHostMemory[m_iInputIndex]指向的一段空间按照每个元素大小为一个float的方式解析成一个矩阵（Mat），其维度为dims_i.d[2]×dims_i.d[3]，像素类型是CV_32FC1（32位浮点数单通道）。
	m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, (char*)m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3]);  // visual studio 不支持对void*的指针直接运算，需要转换成具体类型。
	m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, (char*)m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);

	// 获取类别信息
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
	//m_CudaContext->destroy();  // 首先释放该CUDA上下文使用的CUDA资源
	//m_CudaEngine->destroy();   // 再释放该TensorRT模型的CUDA引擎对象。
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
	sort(input_boxes.begin(), input_boxes.end(), [](DetectionBox a, DetectionBox b) { return a.score > b.score; });  // 降序排列
	std::vector<bool> remove_flags(input_boxes.size(), false);
	auto iou = [](const DetectionBox& box1, const DetectionBox& box2) {
		float xx1 = max(box1.left, box2.left);
		float yy1 = max(box1.top, box2.top);
		float xx2 = min(box1.right, box2.right);
		float yy2 = min(box1.bottom, box2.bottom);
		// 交集
		float w = max(0.0f, xx2 - xx1 + 1);
		float h = max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
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
	// remove_if()函数 remove_if(beg, end, op) // 移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
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

	cv::cvtColor(dstimg, dstimg, cv::COLOR_BGR2RGB);  // 由BGR转成RGB
	cv::Mat m_Normalized;
	dstimg.convertTo(m_Normalized, CV_32FC3, 1 / 255.);
	// 将m_Normalized的每个通道拆分为一个矩阵，然后将这些矩阵按顺序存储到m_InputWrappers的数组中，以替换现有内容。
	cv::split(m_Normalized, m_InputWrappers);  // 通道分离[h,w,3] RGB

	// 创建CUDA流,推理时TensorRT执行通常是异步的，因此将内核排入CUDA流
	cudaStreamCreate(&m_CudaStream);
	auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream);
	auto start = std::chrono::system_clock::now();
	auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);  // TensorRT 执行通常是异步的，因此将内核排入 CUDA 流：
	ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream);  //输出传回给CPU，数据从显存到内存
	ret = cudaStreamSynchronize(m_CudaStream);
}

void YOLOv5::detect_only()
{
	// 创建CUDA流,推理时TensorRT执行通常是异步的，因此将内核排入CUDA流
	cudaStreamCreate(&m_CudaStream);
	auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream);
	auto start = std::chrono::system_clock::now();
	auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);  // TensorRT 执行通常是异步的，因此将内核排入 CUDA 流：
	ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream);  //输出传回给CPU，数据从显存到内存
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

	// 下面的builder, config, network是基本需要的组件
	// 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	// 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	// onnx parser解析器来解析onnx模型
	auto parser = nvonnxparser::createParser(*network, logger);
	if (!parser->parseFromFile(onnx_path.c_str(), 1)) {
		printf("Failed to parse classifier.onnx.\n");
		return false;
	}

	// 设置工作区大小
	printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);

	// 需要通过profile来使得batchsize时动态可变的，这与我们之前导出onnx指定的动态batchsize是对应的
	int maxBatchSize = 10;
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();

	// 设置batchsize的最大/最小/最优值
	input_dims.d[0] = 1;
	input_dims.d[2] = w;
	input_dims.d[3] = w;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);

	// 开始构建tensorrt模型engine
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	if (engine == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	// 将构建好的tensorrt模型engine反序列化（保存成文件）
	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen(build_path.c_str(), "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	// 逆序destory掉指针
	model_data->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();

	printf("Build Done.\n");
	return true;
}