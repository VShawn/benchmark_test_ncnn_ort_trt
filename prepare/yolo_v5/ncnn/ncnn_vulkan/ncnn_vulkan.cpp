#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "net.h"

#if _WIN32
typedef std::wstring path_t;
#define PATHSTR(X) L##X
#else
typedef std::string path_t;
#define PATHSTR(X) X
#endif
#include "wic_image.h"
#include <map>
#include "datareader.h"
#include <filesystem>

using namespace std;


// #define YOLOV5_V60 1 //YOLOv5 v6.0

#if YOLOV5_V60
#define MAX_STRIDE 64
#else
#define MAX_STRIDE 32
class YoloV5Focus : public ncnn::Layer
{
public:
	YoloV5Focus()
	{
		one_blob_only = true;
	}

	virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
	{
		int w = bottom_blob.w;
		int h = bottom_blob.h;
		int channels = bottom_blob.c;

		int outw = w / 2;
		int outh = h / 2;
		int outc = channels * 4;

		top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
		if (top_blob.empty())
			return -100;

#pragma omp parallel for num_threads(opt.num_threads)
		for (int p = 0; p < outc; p++)
		{
			const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
			float* outptr = top_blob.channel(p);

			for (int i = 0; i < outh; i++)
			{
				for (int j = 0; j < outw; j++)
				{
					*outptr = *ptr;

					outptr += 1;
					ptr += 2;
				}

				ptr += w;
			}
		}

		return 0;
	}
};

DEFINE_LAYER_CREATOR(YoloV5Focus)
#endif //YOLOV5_V60





ncnn::Mat load(path_t& image_path)
{
	{
		const path_t& imagepath = image_path;

		int webp = 0;

		unsigned char* pixeldata = 0;
		int w;
		int h;
		int c;

#if _WIN32
		FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
		FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
		if (fp)
		{
			// read whole file
			unsigned char* filedata = 0;
			int length = 0;
			{
				fseek(fp, 0, SEEK_END);
				length = ftell(fp);
				rewind(fp);
				filedata = (unsigned char*)malloc(length);
				if (filedata)
				{
					fread(filedata, 1, length, fp);
				}
				fclose(fp);
			}

			if (filedata)
			{
				//pixeldata = webp_load(filedata, length, &w, &h, &c);
				if (pixeldata)
				{
					webp = 1;
				}
				else
				{
					// not webp, try jpg png etc.
#if _WIN32
					pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else // _WIN32
					pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 0);
					if (pixeldata)
					{
						// stb_image auto channel
						if (c == 1)
						{
							// grayscale -> rgb
							stbi_image_free(pixeldata);
							pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
							c = 3;
						}
						else if (c == 2)
						{
							// grayscale + alpha -> rgba
							stbi_image_free(pixeldata);
							pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 4);
							c = 4;
						}
					}
#endif // _WIN32
				}

				free(filedata);
			}
		}
		if (pixeldata)
		{
			auto inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
			return inimage;
		}
		else
		{
#if _WIN32
			fwprintf(stderr, L"decode image %ls failed\n", imagepath.c_str());
#else // _WIN32
			fprintf(stderr, "decode image %s failed\n", imagepath.c_str());
#endif // _WIN32
		}
	}

	return 0;
}


//这个函数是官方提供的用于打印输出的tensor
void pretty_print(const ncnn::Mat& m)
{
	for (int q = 0; q < m.c; q++)
	{
		const float* ptr = m.channel(q);
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x++)
			{
				printf("%f ", ptr[x]);
			}
			ptr += m.w;
			printf("\n");
		}
		printf("------------------------\n");
	}
}

vector<float> get_output(const ncnn::Mat& m)
{
	vector<float> res;
	for (int q = 0; q < m.c; q++)
	{
		const float* ptr = m.channel(q);
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x++)
			{
				res.push_back(ptr[x]);
			}
			ptr += m.w;
		}
	}
	return res;
}


static std::vector<float> soft_max(const std::vector<float>& cls_scores)
{
	std::vector<float> exps(cls_scores.size(), 0);
	double sum = 0;
	for (size_t i = 0; i < cls_scores.size(); i++)
	{
		exps[i] = exp(cls_scores[i]);
		sum += exps[i];
	}


	for (size_t i = 0; i < cls_scores.size(); i++)
	{
		exps[i] /= sum;
	}

	return exps;
}


static int print_topk(const std::vector<float>& cls_scores, int topk)
{
	// partial sort topk with index
	int size = cls_scores.size();
	std::vector<std::pair<float, int> > vec;
	vec.resize(size);
	for (int i = 0; i < size; i++)
	{
		vec[i] = std::make_pair(cls_scores[i], i);
	}

	std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<float, int> >());

	// print topk and score
	for (int i = 0; i < topk; i++)
	{
		float score = vec[i].first;
		int index = vec[i].second;
		fprintf(stderr, "%d = %f\n", index, score);
	}

	return 0;
}

static int detect_yolov5(const ncnn::Mat& bgr, std::vector<float>& cls_scores)
{
	const int image_size = 640;
	const string param = "yolov5s.param";
	const string bin = "yolov5s.bin";
	std::cout << param.c_str() << std::endl;

	{
		ncnn::Net yolov5;
		yolov5.opt.use_vulkan_compute = true;

		yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
		yolov5.load_param(param.c_str());
		yolov5.load_model(bin.c_str());


		int w = bgr.w;
		int h = bgr.h;
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr, ncnn::Mat::PIXEL_BGR2RGB, w, h, image_size, image_size);

		// transforms.ToTensor(),
		// transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		// R' = (R / 255 - 0.485) / 0.229 = (R - 0.485 * 255) / 0.229 / 255
		// G' = (G / 255 - 0.456) / 0.224 = (G - 0.456 * 255) / 0.224 / 255
		// B' = (B / 255 - 0.406) / 0.225 = (B - 0.406 * 255) / 0.225 / 255
		const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
		const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
		in.substract_mean_normalize(mean_vals, norm_vals);

		ncnn::Extractor ex = yolov5.create_extractor();

		ex.input("images", in);
		ncnn::Mat out;
		ex.extract("801", out);

		cls_scores.resize(out.w);
		for (int j = 0; j < out.w; j++)
		{
			cls_scores[j] = out[j];
		}
		print_topk(cls_scores, 5);
	}

	return 0;
}

std::map<const std::string, const std::string> inputs = {
	{"yolov5n", "images"},
	{"yolov5n6", "images"},
	{"yolov5s", "images"},
	{"yolov5s6", "images"},
	{"yolov5m", "images"},
	{"yolov5m6", "images"},
	{"yolov5l", "images"},
	{"yolov5l6", "images"},
	{"yolov5x", "images"},
	{"yolov5x6", "images"},
};
std::map<const std::string, const std::string> extract = {
	{"yolov5n", "output0"},
	{"yolov5n6", "output0"},
	{"yolov5s", "801"},
	{"yolov5s6", "401"},
	{"yolov5m", "output0"},
	{"yolov5m6", "output0"},
	{"yolov5l", "output0"},
	{"yolov5l6", "output0"},
	{"yolov5x", "output0"},
	{"yolov5x6", "output0"},
};

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
	virtual int scan(const char* format, void* p) const
	{
		return 0;
	}
	virtual size_t read(void* buf, size_t size) const
	{
		memset(buf, 0, size);
		return size;
	}
};

void Test(string modle, ncnn::Mat m, const int input_width, const int input_height, const int loop = 100)
{

	const string extract_layer = extract[modle];
	const string input_layer = inputs[modle];
	const string param = modle + ".param";
	const string bin = modle + ".bin";
	const string layer = "403";

	ncnn::Net yolov5;
	yolov5.opt.use_vulkan_compute = true;
	yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
	yolov5.load_param(param.c_str());
	// 如果文件 bin 存在
	if (std::filesystem::exists(bin))
	{
		yolov5.load_model(bin.c_str());
	}
	else
	{
		// 随机参数初始化
		DataReaderFromEmpty dr;
		yolov5.load_model(dr);
	}


	int w = m.w;
	int h = m.h;
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(m, ncnn::Mat::PIXEL_BGR2RGB, w, h, input_width, input_width);
	// transforms.ToTensor(),
	// transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	// R' = (R / 255 - 0.485) / 0.229 = (R - 0.485 * 255) / 0.229 / 255
	// G' = (G / 255 - 0.456) / 0.224 = (G - 0.456 * 255) / 0.224 / 255
	// B' = (B / 255 - 0.406) / 0.225 = (B - 0.406 * 255) / 0.225 / 255
	const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
	const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
	in.substract_mean_normalize(mean_vals, norm_vals);
	{
		// 预热
		ncnn::Extractor ex = yolov5.create_extractor();
		ex.input(input_layer.c_str(), in);
		ncnn::Mat out;
		ex.extract(layer.c_str(), out);
	}

	clock_t start = clock();
	for (int i = 0; i < loop; ++i)
	{
		ncnn::Extractor ex = yolov5.create_extractor();
		ex.input(input_layer.c_str(), in);
		ncnn::Mat out;
		ex.extract(layer.c_str(), out);
	}
	clock_t end = clock();
	double time2 = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "ncnn " << param << " " << input_width << " time avg - cost " << time2 / loop << " s, " << loop / time2 << " fps" << endl;//输出运行时间
	while (1)
	{
		cout << "当前加载 1 个模型，输入 c 加载第2模型" << std::endl;
		if (std::getchar() == 'c')
			break;
	}


	ncnn::Net yolov5_2;
	yolov5_2.opt.use_vulkan_compute = true;
	yolov5_2.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
	yolov5_2.load_param(param.c_str());
	yolov5_2.load_model(bin.c_str());

	{
		// 预热
		ncnn::Extractor ex = yolov5_2.create_extractor();
		ex.input("images", in);
		ncnn::Mat out;
		ex.extract(layer.c_str(), out);
	}


	while (1)
	{
		cout << "当前加载 2 个模型，且已完成预热，请查看资源消耗, 输入 c 继续后续测试" << std::endl;
		if (std::getchar() == 'c')
			break;
	}
}

//main函数模板
int main()
{
#if _WIN32
	CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

	int gpu_count = ncnn::get_gpu_count();
	std::cout << "gpu device we have : " << gpu_count << std::endl;

	//cv::Mat m = cv::imread("D:/cat.jpg", 1);
	//cv::Mat m = cv::imread("D:\\UritWorks\\AI\\benchmark\\prepare\\cat.1.jpg", 1);
	//cv::Mat m = cv::imread("D:\\UritWorks\\AI\\benchmark\\prepare\\cat.1084.jpg", 1);
	//path_t path = L"D:\\UritWorks\\AI\\benchmark\\prepare\\dog.310.jpg";
	path_t path = L"C:\\dog.310.jpg";
	auto img = load(path);
	if (img.empty())
	{
		fprintf(stderr, "read %s failed\n", path);
		return -1;
	}

	const int loop = 100;
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
	return 0;
}