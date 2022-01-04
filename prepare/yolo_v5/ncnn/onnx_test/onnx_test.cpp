// onnx_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

std::wstring to_wstring(const std::string& str)
{
	unsigned len = str.size() * 2;
	setlocale(LC_CTYPE, "");
	wchar_t* p = new wchar_t[len];
	mbstowcs(p, str.c_str(), len);
	std::wstring wstr(p);
	delete[] p;
	return wstr;
}

Ort::Value create_tensor(const cv::Mat& mat,
	const std::vector<int64_t>& tensor_dims,
	const Ort::MemoryInfo& memory_info_handler,
	std::vector<float>& tensor_value_handler,
	bool target_format_is_CHW)
	throw(std::runtime_error)
{
	const unsigned int rows = mat.rows;
	const unsigned int cols = mat.cols;
	const unsigned int channels = mat.channels();

	cv::Mat mat_ref;
	if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
	else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

	if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
	if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

	// CXHXW
	if (target_format_is_CHW)
	{
		const unsigned int target_height = tensor_dims.at(2);
		const unsigned int target_width = tensor_dims.at(3);
		const unsigned int target_channel = tensor_dims.at(1);
		const unsigned int target_tensor_size = target_channel * target_height * target_width;
		if (target_channel != channels) throw std::runtime_error("channel mismatch.");

		tensor_value_handler.resize(target_tensor_size);

		cv::Mat resize_mat_ref;
		if (target_height != rows || target_width != cols)
			cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
		else resize_mat_ref = mat_ref; // reference only. zero-time cost.

		std::vector<cv::Mat> mat_channels;
		cv::split(resize_mat_ref, mat_channels);
		// CXHXW
		for (unsigned int i = 0; i < channels; ++i)
			std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
				mat_channels.at(i).data, target_height * target_width * sizeof(float));

		return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
			target_tensor_size, tensor_dims.data(),
			tensor_dims.size());
	}
	else
	{
		// HXWXC
		const unsigned int target_height = tensor_dims.at(1);
		const unsigned int target_width = tensor_dims.at(2);
		const unsigned int target_channel = tensor_dims.at(3);
		const unsigned int target_tensor_size = target_channel * target_height * target_width;
		if (target_channel != channels) throw std::runtime_error("channel mismatch!");
		tensor_value_handler.resize(target_tensor_size);

		cv::Mat resize_mat_ref;
		if (target_height != rows || target_width != cols)
			cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
		else resize_mat_ref = mat_ref; // reference only. zero-time cost.

		std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

		return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
			target_tensor_size, tensor_dims.data(),
			tensor_dims.size());
	}
}


Ort::Value create_tensorfp16(const cv::Mat& mat,
	const std::vector<int64_t>& tensor_dims,
	const Ort::MemoryInfo& memory_info_handler,
	std::vector<float>& tensor_value_handler,
	bool target_format_is_CHW)
	throw(std::runtime_error)
{
	const unsigned int rows = mat.rows;
	const unsigned int cols = mat.cols;
	const unsigned int channels = mat.channels();

	cv::Mat mat_ref;
	if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
	else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

	if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
	if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

	// CXHXW
	if (target_format_is_CHW)
	{
		const unsigned int target_height = tensor_dims.at(2);
		const unsigned int target_width = tensor_dims.at(3);
		const unsigned int target_channel = tensor_dims.at(1);
		const unsigned int target_tensor_size = target_channel * target_height * target_width;
		if (target_channel != channels) throw std::runtime_error("channel mismatch.");

		tensor_value_handler.resize(target_tensor_size);

		cv::Mat resize_mat_ref;
		if (target_height != rows || target_width != cols)
			cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
		else resize_mat_ref = mat_ref; // reference only. zero-time cost.

		std::vector<cv::Mat> mat_channels;
		cv::split(resize_mat_ref, mat_channels);
		// CXHXW
		for (unsigned int i = 0; i < channels; ++i)
			std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
				mat_channels.at(i).data, target_height * target_width * sizeof(float));

		std::vector<uint16_t> inputTensorValueFp16;
		for (auto fp32 : tensor_value_handler)
		{
			inputTensorValueFp16.push_back(0);
			//inputTensorValueFp16.push_back(float32_to_float16(fp32));
		}
		return Ort::Value::CreateTensor(memory_info_handler, inputTensorValueFp16.data(),
			inputTensorValueFp16.size() * sizeof(uint16_t), tensor_dims.data(),
			tensor_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	}
	else
	{
		// HXWXC
		const unsigned int target_height = tensor_dims.at(1);
		const unsigned int target_width = tensor_dims.at(2);
		const unsigned int target_channel = tensor_dims.at(3);
		const unsigned int target_tensor_size = target_channel * target_height * target_width;
		if (target_channel != channels) throw std::runtime_error("channel mismatch!");
		tensor_value_handler.resize(target_tensor_size);

		cv::Mat resize_mat_ref;
		if (target_height != rows || target_width != cols)
			cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
		else resize_mat_ref = mat_ref; // reference only. zero-time cost.

		std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

		//return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
		//    target_tensor_size, tensor_dims.data(),
		//    tensor_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

		std::vector<uint16_t> inputTensorValueFp16;
		for (auto fp32 : tensor_value_handler)
		{
			inputTensorValueFp16.push_back(0);
			//inputTensorValueFp16.push_back(float32_to_float16(fp32));
		}
		return Ort::Value::CreateTensor(memory_info_handler, inputTensorValueFp16.data(),
			inputTensorValueFp16.size() * sizeof(uint16_t), tensor_dims.data(),
			tensor_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	}
}

void Test(std::string _onnx_path, cv::Mat img, const int input_width, const int input_height,
	const int loop = 100,
	bool useGpu = true, int num_threads = 1)
{
	_onnx_path = _onnx_path + ".onnx";
	std::string input_node_name_str = "images";
	std::vector<int64_t> input_node_dims{ 1, 3, input_width, input_height }; // 1 input only.
	std::size_t input_tensor_size = input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3];  
	std::vector<std::string> output_node_name_strs{ "output0" };
	std::vector<std::vector<int64_t>> output_node_dims{ {1, 25200, 85} }; // >=1 outputs


	auto ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Test");
	// 0. session options
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(num_threads);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetLogSeverityLevel(4);
	// GPU compatiable.
	if (useGpu)
	{
		std::cout << "useGpu GpuId = 0" << std::endl;
		OrtCUDAProviderOptions provider_options;
		session_options.AppendExecutionProvider_CUDA(provider_options);
		//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
	}

	const wchar_t* onnx_path = nullptr; // 不支持中文路径
#ifdef _WIN32
	std::wstring _w_onnx_path(to_wstring(_onnx_path));
	onnx_path = _w_onnx_path.data();
#else
	onnx_path = _onnx_path.data();
#endif
	std::cout << input_node_name_str << " " << input_node_dims[0] << " " << input_node_dims[1] << " " << input_node_dims[2] << " " << input_node_dims[3] << std::endl;

	// 1. session
	auto ort_session = new Ort::Session(ort_env, onnx_path, session_options);

	//// 2. get input shape
	//{
	//	const size_t num_input_nodes = ort_session->GetInputCount();
	//	assert(num_input_nodes == 1, "input number should be 1!");
	//	Ort::AllocatorWithDefaultOptions allocator;
	//	auto input_name = ort_session->GetInputNameAllocated(0, allocator);
	//	auto type_info = ort_session->GetInputTypeInfo(0);
	//	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	//	input_node_name_str = input_name.get();
	//	input_node_dims = tensor_info.GetShape();
	//	std::cout << input_node_name_str << " " << input_node_dims[0] << " " << input_node_dims[1] << " " << input_node_dims[2] << " " << input_node_dims[3] << std::endl;
	//}

	//// 3. 
	//{
	//	output_node_name_strs.clear();
	//	output_node_dims.clear();
	//	const size_t num_output_nodes = ort_session->GetOutputCount();
	//	Ort::AllocatorWithDefaultOptions allocator;
	//	// iterate over all input nodes
	//	for (size_t i = 0; i < num_output_nodes; i++)
	//	{
	//		auto name = ort_session->GetOutputNameAllocated(i, allocator);
	//		auto type_info = ort_session->GetOutputTypeInfo(i);
	//		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	//		auto dims = tensor_info.GetShape();
	//		output_node_name_strs.push_back(name.get());
	//		output_node_dims.push_back(dims);
	//		std::cout << name << " " << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << std::endl;
	//	}
	//}


	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;

	// 4. 
	{
		input_node_names.clear();
		input_node_names.push_back(input_node_name_str.c_str());

		input_tensor_size = 1;
		for (unsigned int i = 0; i < input_node_dims.size(); ++i)
			input_tensor_size *= input_node_dims[i];

		output_node_names.clear();
		for (size_t i = 0; i < output_node_name_strs.size(); i++)
		{
			output_node_names.push_back(output_node_name_strs[i].c_str());
		}
	}


	cv::Mat mat_rs;
	cv::resize(img, mat_rs, cv::Size(input_width, input_height));
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<float> input_values_handler;
	Ort::Value input_tensor = create_tensor(mat_rs, input_node_dims, memory_info_handler, input_values_handler, true);
	//Ort::Value input_tensor = create_tensorfp16(mat_rs, input_node_dims, memory_info_handler, input_values_handler, true);


	{
		// 2. inference scores & boxes.
		auto output_tensors = ort_session->Run(
			Ort::RunOptions{ nullptr }, input_node_names.data(),
			&input_tensor, 1, output_node_names.data(), output_node_names.size()
		);
	}



	clock_t start = clock();
	for (int i = 0; i < loop; ++i)
	{
		auto output_tensors = ort_session->Run(
			Ort::RunOptions{ nullptr }, input_node_names.data(),
			&input_tensor, 1, output_node_names.data(), output_node_names.size()
		);
	}
	clock_t end = clock();
	double time2 = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "onnx " << _onnx_path << " " << input_height << " time avg - cost " << time2 / loop << " s, " << loop / time2 << " fps" << std::endl;//输出运行时间
	while (1)
	{
		std::cout << "当前加载 1 个模型，输入 c 加载第2模型" << std::endl;
		if (std::getchar() == 'c')
			break;
	}

	{

		// 1. session
		auto ort_session2 = new Ort::Session(ort_env, onnx_path, session_options);
		std::string input_node_name_str = "inputs";
		std::vector<int64_t> input_node_dims{ 1, 3, 244, 244 }; // 1 input only.
		std::size_t input_tensor_size = 1; // = input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3];  
		std::vector<std::string> output_node_name_strs;
		std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
		std::vector<const char*> input_node_names;
		std::vector<const char*> output_node_names;

		// 2. get input shape
		{
			const size_t num_input_nodes = ort_session2->GetInputCount();
			assert(num_input_nodes == 1, "input number should be 1!");
			Ort::AllocatorWithDefaultOptions allocator;
			auto input_name = ort_session2->GetInputNameAllocated(0, allocator);
			auto type_info = ort_session2->GetInputTypeInfo(0);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			input_node_name_str = input_name.get();
			input_node_dims = tensor_info.GetShape();
		}

		// 3. 
		{
			output_node_name_strs.clear();
			output_node_dims.clear();
			const size_t num_output_nodes = ort_session2->GetOutputCount();
			Ort::AllocatorWithDefaultOptions allocator;
			// iterate over all input nodes
			for (size_t i = 0; i < num_output_nodes; i++)
			{
				auto name = ort_session2->GetOutputNameAllocated(i, allocator);
				auto type_info = ort_session2->GetOutputTypeInfo(i);
				auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
				auto dims = tensor_info.GetShape();
				output_node_name_strs.push_back(name.get());
				output_node_dims.push_back(dims);
			}
		}

		// 4. 
		{
			input_node_names.clear();
			input_node_names.push_back(input_node_name_str.c_str());

			input_tensor_size = 1;
			for (unsigned int i = 0; i < input_node_dims.size(); ++i)
				input_tensor_size *= input_node_dims.at(i);

			output_node_names.clear();
			for (size_t i = 0; i < output_node_name_strs.size(); i++)
			{
				output_node_names.push_back(output_node_name_strs[i].c_str());
			}
		}

		{
			// 2. inference scores & boxes.
			auto output_tensors = ort_session2->Run(
				Ort::RunOptions{ nullptr }, input_node_names.data(),
				&input_tensor, 1, output_node_names.data(), output_node_names.size()
			);
		}



		while (1)
		{
			std::cout << "当前加载 2 个模型，且已完成预热，请查看资源消耗, 输入 c 继续后续测试" << std::endl;
			if (std::getchar() == 'c')
				break;
		}

		if (ort_session2)
			delete ort_session2;
		ort_session2 = nullptr;
	}

	if (ort_session)
		delete ort_session;
	ort_session = nullptr;
}

int main()
{
	std::cout << "Hello World! \n";
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