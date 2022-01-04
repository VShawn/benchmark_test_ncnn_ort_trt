#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "net.h"

#include<io.h>

#if _WIN32
typedef std::wstring path_t;
#define PATHSTR(X) L##X
#else
typedef std::string path_t;
#define PATHSTR(X) X
#endif
#include "wic_image.h"

using namespace std;

wstring string2wstring(string str)
{
	wstring result;
	//获取缓冲区大小，并申请空间，缓冲区大小按字符计算  
	int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), NULL, 0);
	TCHAR* buffer = new TCHAR[len + 1];
	//多字节编码转换成宽字节编码  
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), buffer, len);
	buffer[len] = '\0';             //添加字符串结尾  
	//删除缓冲区并返回值  
	result.append(buffer);
	delete[] buffer;
	return result;
}


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




static int detect_mobile_v3(const ncnn::Mat& bgr, std::vector<float>& cls_scores)
{
	{
		ncnn::Net mobilev3;
		mobilev3.opt.use_vulkan_compute = true;

		mobilev3.load_param("mobilenetv3-large-1cd25616.pth.sim.param");
		mobilev3.load_model("mobilenetv3-large-1cd25616.pth.sim.bin");


		int w = bgr.w;
		int h = bgr.h;
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr, ncnn::Mat::PIXEL_BGR2RGB, w, h, 224, 224);

		// transforms.ToTensor(),
		// transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		// R' = (R / 255 - 0.485) / 0.229 = (R - 0.485 * 255) / 0.229 / 255
		// G' = (G / 255 - 0.456) / 0.224 = (G - 0.456 * 255) / 0.224 / 255
		// B' = (B / 255 - 0.406) / 0.225 = (B - 0.406 * 255) / 0.225 / 255
		const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
		const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
		in.substract_mean_normalize(mean_vals, norm_vals);

		ncnn::Extractor ex = mobilev3.create_extractor();

		ex.input("input", in);
		ncnn::Mat out;
		ex.extract("output", out);

		cls_scores.resize(out.w);
		for (int j = 0; j < out.w; j++)
		{
			cls_scores[j] = out[j];
		}
	}

	return 0;
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
	auto m = load(path);
	if (m.empty())
	{
		fprintf(stderr, "read %s failed\n", path);
		return -1;
	}

	//{
	//    std::vector<float> cls_scores;
	//    std::cout << "detect_mobile_v3" << std::endl;
	//    detect_mobile_v3(m, cls_scores);
	//    auto p = soft_max(cls_scores);
	//    print_topk(p, 5);
	//}




	{
		string files[] =
{
			"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\00bdf4c3f30d423429ecfbca3df222a9_23_1562_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\012eba957ba15c516162b1d42db3eaa6_416_1832_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\01325ea5ba4c401dd080bb7264eb3f4c_620_386_1559.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\01383cbdbd87925d3f2bf45aa74a3220_84_1348_1561.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\014ae197f3934800dc710daf5a68cc24_5_1952_440.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\01596829ce2dfed626348d71cb1eb57e_29_1996_344.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\01b88a11f930dedadb657fb595fa03f2_514_10_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\01c7f03b61ad74226854dbfb94695783_281_4_1514.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\01dfc2a98f6a34c5c7a7aee9642605a6_178_1856_1186.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\01e3c81fd651c492723d206921702fa5_173_1944_1216.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0208650f7bba3ef35a1feb1f0a6c26cd_273_240_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\02182f6e6884775d9be41d2191f0947a_365_1344_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\029a9ef1b8b23e79a0d4324f298d7782_176_4_872.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\02fe10203d07f8906507656725a85b14_440_4_1072.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\02fec03ee0c6296c9e55e1f41d427058_172_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\032c94a292fc9765c122b702e2172b96_519_600_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\032cb625d774039c8a7e77c0d71e0e00_550_1994_686.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0332991cf3c690a782014de3fb97997c_284_1952_11.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0343c4862fdae80b78b39c66abc39c66_468_1108_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\03575377a3318755d5e7dcfd47a71edb_219_482_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\035f1c18e69f855ced277482f3ff5401_527_1386_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\03894acff30ad7b9e91fbe6341eb77f5_779_1232_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\038a604c0bd44be9faf7f1ace9b45966_136_2004_460.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\03bc8566e86c683a6e31274c244a7144_243_1990_658.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0458ad86bef9d614685aa7de89f8d6df_736_4_742.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\047ec041d5f1661ffc40c60ac9ef0b61_236_236_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\04afdf1b8f4d03aefedc61944b5173ea_539_298_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\04d8d6da4998c03c4d03178c33986893_333_4_572.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\051c0d95edda27e2e1b6b16b903bcb1c_99_1984_339.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\055dbabf07dee9e5536fbd401c1fb7f9_222_1950_352.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\05874490e0c07b76724e3a62248d5f75_26_4_1002.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\064b22b7b3e92c2acbf5ecf56f5179e4_512_454_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\069a995d0c1636a67b44fe7a72bd50df_625_4_678.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\069fe0aea3e28b3bbdb83253dac62d78_136_1648_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\06c742383bd02bfa4f2733c204b615d7_327_4_1564.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\06eb8d37a6623024dca2b1ce5b65ef4e_282_1742_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\070c2723df39156c0fd0f115c2f9c5f7_642_1616_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\075254bbee91897e0bce77e72cc80884_3_4_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\076a0e128eaac7ec93b736a70a4bf428_382_4_1697.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\07a1effe22b322270411b99592e8be8e_682_4_598.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\07be4c49e3780b72df845f1572452d2e_223_1916_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\07c435bc989d5fc5e66115d1ae458d6c_759_4_1122.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\07d205ad7f88593acf108571a35044e3_162_1982_444.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\07fb403d7dbaeb34b617edb10071df37_313_774_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\080d360f14ccbcd2db77bda327269aa5_263_1514_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\083099ababcec748fcc2ef51cbbe5158_511_1076_1733.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\086229310d73f1629833cb480b45e4f6_378_2008_2007.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0889d7e40382a63e06a76a2b5b235e59_511_192_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\097ba2319c497333464ccf8dda6800d1_501_1902_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\09c2b70398143ed8f978de8ade70695e_158_4_1534.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\09fc1f92c831eeb7688ac4fc02d6bf83_392_794_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0a13ce59731299477dd773b4d094a243_182_4_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0a9799737e4b4f8403c3e7c1b85b7299_588_4_592.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0a97c7e014eb3159f220b4ddf5c6d0ad_475_1980_1442.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0acf9d77ae6eea63db93f2ede4aa1266_162_1406_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0ad9087bb0116cd409a0bf9968e94538_107_1990_1698.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0b3cd2c106fb72cc8d8bc7740dbd15e2_557_536_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0b4a4e6cefa378cd87fbd0d4c6124b9b_399_1600_1440.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0bd77f9039fadc9b35df392f448dc877_231_4_774.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0be8740c54f279544c7690af407cb3b6_18_4_1090.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0bf0fe70602014f92f31896ebbcc6597_793_2000_948.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0bf29edde2bbed729c61abac045ee886_172_4_726.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0c03f13eee73ce8c799f6e47febf6474_113_4_160.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0c491878dc3f9235a2b3f98c4f0be3eb_210_1956_33.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0c64890f813ef913268d9aada69fff1c_505_384_944.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0c7de5bfa2d9d5a407c567f08f5db0c3_422_1968_905.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0c8eeae80453657ee05c3e72c0ca06e2_777_2008_457.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0ccf4f82dda62c27068afaa507c3de35_452_1482_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0cd1e8fc4bd526452f76d8f02907b840_174_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0d0e319f6e7fa4fdbfba6ffc33e510f2_127_1956_299.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0d1e5cb092b0fdfc0e4d8d8dd07f5276_780_1984_1014.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0d7527d4b32ce588e1cb24363fcfe2b1_461_638_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0da616bf560b1b05b68934bc1c6903d8_352_1302_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0debfa7ecfd0e2695e9b8b5c514cbe87_430_1356_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0e5f8e681329dea7c500be5e1dcd7d15_234_1762_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0e9977ced86a6ca0bdeb6c8b70229f63_1_4_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0e9e0d8f08f8657aea465c71bead861a_331_1932_1212.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0eefd430aadb8d45b4eeb4b173a2ffec_369_0_1678.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0efe0a0302cd7f3d5c1d7bc69248e2bf_117_1952_155.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0f304274ac0bec452504da7be9b0d2f1_518_1984_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0f3af3a19523a2b7eb1ed59c90558a3e_208_1880_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0f4296e0c798b68ce101423002aeac63_552_4_216.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0f4df040e920e146935ef60a9e69a4b6_314_4_1878.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0f802119c86e54ab399337006ffc4f77_504_0_1740.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0f9dd766b34db300aa970242fabc639d_492_0_1499.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\0ffd259a18ca4c48f2c0c5b256cc6e2b_433_1038_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\106cce38284c69e0a3f493bd2bf2dd67_731_1274_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\107b61700e0ace2db0588360a368f389_201_706_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\10a6d0c01818ba7e4a1ace816a90aa76_666_4_168.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\10ee71ca953e56c7de1767dcc93d0fef_567_4_1186.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\111cb78a920cdd637451516021ae5eb0_37_4_1288.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1181e213c1fe16d916535b325f83755d_320_1960_263.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\118db628e02e531d14dc92d9c9009311_420_0_1160.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\118fbe669e93d485bc23dbb2f5edc566_101_4_1808.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\11a9d8b7ad326dac91fd7c9e80902fe4_176_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\121d0ed8bbb627d3cc2228a1273d1872_413_4_1776.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\122ebd0cbf2383dd805726ddd73958c7_12_1304_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\124146176a8d61a6089989b666e198ec_463_216_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\124ebda96284f1b543f391762dd702cc_383_4_980.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\12944c3bdc3f3fbf6852e0777501b24b_478_200_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\12d3d76c9ddbf75357ba40f0e1ae4a91_698_1960_1930.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\12df78626a1ad4b1d44bf255c454f8c1_216_1976_561.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\12ed670ae6b1aa367fddabaae1f79194_510_370_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\13063458abfe4f08cf948d78294f2a8b_278_1742_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\131898d3cf20d341ae1c869046cce724_142_1996_1643.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\13240b0853beacb4ce6e61c046b657a2_483_592_1077.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\13e65c379abcf77512572cf30b4388da_613_4_600.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\13f0273790d1dfc36d99a501aedf76cc_453_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\145b03578d0cb9ac2c146fdb8c0b0540_607_0_1924.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1467f0fa1070c1dff8bbe94bee9989c6_527_4_118.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\14d5d18fc833f6d93c92012226378778_34_0_78.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\14f542a60928104364d885418fd26a8f_275_790_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\150642c1beb7a68363289579c0e1b0c8_0_1806_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\159787d839d0285d5f4094c89a173971_116_1594_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\15e55d1f95ab044fac4eadd9f4d8635c_432_1952_1939.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\161801e2b2676189c74d40e54bf48e61_40_1308_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\161ee772ba79965649aed346da7aa9d8_317_696_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\162e57b7c87a4c07ce613dff8c17ace5_582_2000_254.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\16331b32c2ebde997f587a488b0a869c_775_1982_294.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\16596bd9c7a4bc18022b41857fd8216a_484_4_1500.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\167cc5596801caf2c8dd380dd825db26_488_1420_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1688a22facd054b3eeb407fe7bac057c_656_4_938.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\16979c2369ababb192a2c25b58973d3a_708_904_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\176413769e15ae414e232995bd8003ba_622_262_1152.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\18b04a13599b1ecb8ff6f8b66b7ab6be_8_1422_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\190ee045653ae29b81314aca27a88d25_319_1986_288.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\196c1cd9e3d4556f49a7c1f656a3b97c_526_544_458.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\19fb86258b4d07af794654837d20489b_196_1932_1388.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1a3442e8540d391bb5d237b227a8153d_240_4_458.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1a8a7d944a6210ac94aaf73ea095934b_716_1992_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1af307343469aa83327393ffa9b18abd_90_1986_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1b0a9a53d6838e97bb32a92b46a73467_242_1372_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1b28c4badbe8c5b99960c67c79c74aba_356_4_978.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1b4c4b4f43a41cee5a178251f905a949_544_1728_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1bc3f0e8a13b688ae55906218484626c_734_1960_632.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1bc94233c71883c659459ee3c50d9d02_574_4_546.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1cd81b87a100b098366603b8af416e8d_6_1142_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1ce6ee92baec2e6fa39a5bf3034fb10e_383_1920_108.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1d6cd24541fc783a6d46bd567dfa73d4_356_66_96.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1e2508c0d824812817ccfbe28d50cccf_731_4_924.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1e3a508e0dfabf9b80d022140663fa09_128_4_798.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1e3f68194970529e0ef2348b1983c4ac_709_1406_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1ea069a51d4ec44d2990f912350b8e3b_563_146_1026.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1edfafac657bcedb2200ce898b966722_297_1956_866.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1f3a5c90e6be7221536b172ad1c6dd38_95_16_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1f6a2c5822e8cfd2884f12b08533b9c0_8_1956_883.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\1f70b6e8fd3653c825dd288e4c8a9b9d_223_1512_295.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2012ff78368ebc0e5698919336036421_696_4_124.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\201a56d31084a439409e08b78707ef1d_581_420_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\20bde7af6c2b585bf7bbc8bd9a2c7f2f_797_0_1604.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\20e4c2c96e8045feb354415727b979e7_370_410_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2109fd4475100adf746070b5bf790d76_81_4_960.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\215a639ea2e9b09368d9019ef9725d70_131_1420_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2164e90a68dccd2edf4242b8b0a6c61a_739_4_1296.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\216a7fdcc9a7dc64e6669a4cebd26ac0_65_1808_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\219347c715aa8d429aa350981105bee9_387_1766_508.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\21c2afcf14290985653fabe7c06b505d_4_1866_1246.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\22095c08796a1db0515925c3c979ccd6_247_1934_64.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\220e98f2aa27cd855d4df1b3e9e813f7_654_4_316.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2220f86c5f8528329da339edcdf62629_120_1514_900.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\228becc99a123ec598bf5339ff77fac6_259_1776_1728.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\23217bb068c732ff11fa69da8fe2fe62_450_4_1892.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\239c7002b7eff544f2ade3bb9ba3e313_774_1964_1650.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\23d161f2f74c9954a2c4d5f69cfa625b_255_1378_1651.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\23e6659b4c04f82ab9f10f9c4cf556ea_376_1154_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2413b8cd12ae38b9611ab3b308a24504_197_1200_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2433674c115259f204e07accc069f530_307_516_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\24431fbff37c95efcc705897be079e1a_168_1054_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2491f6df0b927f4513b06c0213702dd9_81_1610_1008.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\24927c914c7bfa22cc95f260c191fb6a_191_402_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\249af1dd107cfc95e93f7e91a6a179a5_174_368_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\249f61202069f1b0d6266a3df89b9fa5_246_1938_1492.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\24b7a7e1a6719271c266736861096cf2_575_156_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\255547ebeff70a9d8b08c547ca952bb3_459_4_508.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\25d9489f77e63afa0472e36beb328a33_112_4_1496.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2648ee1a3dea4c63e9caaf45b624be34_730_90_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\26692be6a7990890f8062b0f78652bf9_387_0_1444.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\268083d31dbed338a5a1e76cacff29a4_578_1998_942.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\26b6ef329053368afa6452c7983d6395_300_1624_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\26e252283120594d7696ac60bc6a0cb7_368_40_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\26f683e1613f74ef64b4f606f3d31b5b_784_0_1505.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2710cc2cac9ad1be7efa9b84d6e23c33_691_4_1208.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2712af26e85ee9fc414e121070815d6a_244_0_1909.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\271358bc3433e53a3678416a983ab3e4_227_608_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2798d05b56d8b306fdd8d8ad05ef13cd_710_252_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\27b36fd7e6fca214b32a1ff8baa03075_735_0_869.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\285fc5e61e2b34177d8b891e6000b9fa_630_4_1368.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\28c1fa5e7a01873f203bb5c83f098721_33_1996_542.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2905ba21a4f1f525aca5fa40ed54c546_390_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2999f8ac9edb6476d2c66f3c2c04d973_609_1966_1112.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\299ebf6dd1bd716f6e289314ff9c5a9a_509_1964_895.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\29e8464446e53532c203df4d004fcdc2_203_1894_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2a268cce39eb5ca718423a0e6c096dfa_185_24_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2a2e596fffe9782ce590e1f37a1bf112_642_4_720.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2a55a462d8e4a693669ed83b0782dca5_311_1186_242.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2a88a00f764a9f22397d08770dc9e529_410_1610_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2a97efbc0e89840b4fbf1f2bfdda0d6e_38_1976_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2abc2123d1af8a86a710bb113bca798c_567_1948_796.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2acd917b2bb621ea21b0420aab272acf_359_4_658.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2af4d6cc85c1a7659c4b1216ed9e2f94_105_0_800.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2b13776446cdef7636dfead5e78682d2_368_1920_832.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2b1533b4291242b1977d0f78acbaac83_297_4_834.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2b8c7817f79d88e0bc9174e4aaf2fd39_687_4_378.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2c0237e19da35cdc9f58a27b78a7235f_225_40_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2c13e31535828c3f4617c8892470eda5_101_1978_434.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2ca73ed509caa4050f6ed4ebdda511d7_482_0_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2cac71f5388adada25fd125492fd7e28_15_906_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2d23bda488b0afebc10f3dbeb0dfa15f_305_1988_1076.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2d50033bece6c9f6b8978479d5f75965_697_1972_1232.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2d5ceb735efdb06c4aefd68ed972d77b_543_838_670.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2d7734f457160ab58b6a0d70df62cd47_529_382_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2d9151d2d23a94e349495455bb7756ad_73_1866_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2d974ab50b23457ad4424ef1d74336de_209_0_493.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2db4c294730b1d79caa7f59d800c9dc5_4_1364_458.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2dc1601b353d21ad33cab39d61e1daba_163_154_936.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2e3ae7465d3514f6253a429cdfd7d3de_126_276_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2e5ed2ba9b79c3b6010cd2079a7c2590_87_0_535.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2ebea14968fb4a877e9ed3a1f686fc95_593_186_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2ee8449b8d63669241e080bcc7db8752_103_1756_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\2fc21524cc36aa6ccaef27e393a804d5_272_1948_440.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\302e1a0652248710f514b1fcd1e4b23e_409_0_1890.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\30510e9dcf8fde7baa0e584994f18346_341_4_1202.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3057e654d4d52b8e396566e06612b54b_357_990_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\30d27e6d778e861f78bbca67257f83f7_791_1942_1476.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\30f58ca7625d46c75056a8ec39a40ec7_702_1958_42.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\31019400635d2102929507b02061c194_490_156_1713.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\310217551848a035e10af865d72ea76e_145_1012_1382.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3117e3ed8ea01a192fb9628a0ba1fb74_81_1288_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3120d182b5a6bf3fae742af333f0b233_568_1930_1624.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\318f2348319629b3791fce564b7f4e36_625_4_1576.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\31aeb5d3eb6facee0a0950ea6cfd838e_676_1974_1344.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\31b81ee6f49ed6ed22f89229ab7b0b60_41_4_1526.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\31e77529671149bd7d66feaaaf8aaa36_649_4_102.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\31f9aab3cc64c315d18995ca0bb21ffd_506_290_538.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3221ce787a57ae29fa7e284d60392c5d_146_338_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\32261b4e404e810e74aa2f3020d1252f_433_1886_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\32aba2f0b2b548ec54e8d77082401de1_389_78_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\32c5b117225ceea20284c8794a64c6ae_675_4_1230.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3327a340baf712c5b09bfceee802cb39_547_1328_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\334cf5388c418ea4d332d1bf1a2b9ed4_227_4_1250.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3358858f430dec38954065885df2e858_97_4_36.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\33c6ff93acffc4a51d61be1b4804978b_756_1990_176.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\33db05b8b3eb45199ec41aba693f5593_275_1808_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\340837fc4971739cfaaafea58bbb9af8_275_0_497.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\346e7f65e68a3789121ba9ea69012fea_472_1752_1363.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3472b2bbc89b94f66bb263c4968e5d7e_523_832_845.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\34925670d4ac2a8de4314490e3518862_29_838_252.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3495b89d1a3a8ed1f7fee36ff8b3e19c_117_1824_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\34c66e594609437403f21854edac7dea_22_0_88.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3518e55fb620b0b0d19fa479e0354dbc_279_1632_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\357feea8304eae3113c23bdd6f95aa95_6_0_1940.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\35a2638d320c4a4a97216ae3761ac92b_640_1968_503.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\35b3dabfd2485ef54e9ce471299528e6_85_2008_1162.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\35e04bd717426ef238d68bb381c406b8_44_1972_896.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\35e4d10ae1e75a4ae5326f5a8bb68891_733_1102_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\366be05d3d44efc7197dcf16f246b5a0_189_2004_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\367ad10231cc6510f24b7fcbacc86128_519_1952_1554.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\36c145631dca739410771490643639a9_82_4_560.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\36ec8319de045614b84fe8e3410cb753_268_1566_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\36f4c6a41cd9524707653e2a85d10486_687_1976_1556.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\37063e1aca325c92529f27d024a96659_572_1956_972.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\374bdda46961584ccf3dca67e68abb0f_316_1984_403.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\374d467721adb51cae390eae52dc858d_634_724_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\37d363e899dc29393d8d956050ab2585_351_1956_1016.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\383ac60eac6a633869cd53cbf23a11da_40_1932_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\38805a2c14d2de774d167e645ed859e7_201_1562_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3884c5f5926adba6d5ff3771e95d73e8_230_1638_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\389356761211f5eae24cc88c2918f4a0_594_956_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3895dae5780ece5f09aa8feeced56945_446_1646_502.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3895e76205d5804df7c4871e0c34daa2_448_96_1396.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3897257be774d9d904d3bccbe62877cf_517_4_1329.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\38b79932e53a31ed0377a1b7f3fbe4b3_633_1952_133.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\38bb5d191185e336feb4d089ac51e79c_231_1956_63.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\38e117671e5259b69d956ed8730f833f_1_4_1046.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\390d4c921400253e4ec732120ee20124_120_0_8.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\397c021f36c646cbe905d70bee33bcf9_793_1926_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\39e0b0b35ca6d337c0dc35326860d5db_186_1708_1866.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3a77c6e957e39d6e928dcb2c8110f7c5_153_570_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3b0c93873261bca3c4fc789feb1ce182_26_1100_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3b3234f85a220470dcdef4f592a76042_73_1096_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3b98662e5dc083a01ea6d001640b4182_737_1952_1710.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3bd5e75efea81f29b303de5d5dbb5a4b_136_556_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3be1eaff2601a86bcb49610b483bdd39_331_1104_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3c9ee63d295848b855d67ebe87a1319f_286_0_419.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3ca6c177c7a7e8b86de6c5baf8930c95_293_0_1923.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3cbf11911c5b4ee8e8d6f8db01aa935c_366_902_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3cfffa35c41e4e710ae327d2faf2bc92_185_1898_572.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3d09b9fd175c09785fc65cc958105644_149_660_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3d32f6318d38fe98b1319fdad7645668_235_4_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3d82d8776e69b83e983b4e8ca0ccd06b_567_880_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3dae44f5aab385d54d7f0f53385e6e47_714_1960_1142.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3fa608b5c68f5300a8161f4783cd9714_63_1992_487.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3fb3f1b637b8dc062bcce4e292312ec8_612_4_120.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\3fdc63980479d33cadba07f6b537a506_106_4_1380.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\402ddcad08b50fb9db6ef1e44a25e011_101_1330_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\404110dbe81550080f9f4a3c65af0fc6_308_1418_1669.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\404fec12b5de6b11997a47b008a04689_82_4_1814.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\406c194078ef58474409ad1870ba17c6_44_1296_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4074f4ab75ec1ac6a3effffbe039db74_75_1890_124.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\40f2ce1c9c5d1c36f9ed0d6a7aa28aef_263_776_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\41084fcb7b7ff187fcc3aef9301147b2_287_4_456.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\410f8cda5650612b8a99c3acfc399a54_184_1984_1337.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\411e9b5e0b9da0bd48d315953e579154_519_4_462.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\412a10a2f29d1b070aeb483873460839_594_422_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\414b9cac142ec02b39f732341c3af9de_767_800_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\41d0f058865fd28dfc27a72ea69879ef_292_462_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\41d1d3b04b716f7e74b5800fb6290384_542_1714_1967.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\42082f2aed5ffa04aa345be123bb484b_341_4_1390.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\421077bec062f03a4c0500de75a1a2a5_663_1988_454.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4250926f8bf35412e8f803808f508e6d_277_506_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\42574b6560f952de1ca47e4c12c2f8d2_645_1750_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\425a96063aba64c615fbded8eb1f936a_533_1946_280.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4273fa3444b04421d63b24f819c428c1_509_4_480.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\42931290359536304da17e0914c1105f_700_1964_1178.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4298a750e11e3d06b76c492853144c82_621_4_522.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\42cf02b9450cf60a3ede4bf9753add31_756_574_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\42d2a57083b57881bcce699f241ceca1_242_280_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\430e4d8d21c08b3bc4440f52d0a1b092_614_1622_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4321e4bdb2661935dabf6500a0ff0193_308_36_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4404bc36ac55674f3aed3d245ba2cb14_494_4_932.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\441b255dc7d48575a906426f4cbab336_20_1890_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\443842d349b19f7e23ef0509cd2e2a2e_255_20_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\445e1cbdc2f6b175b7ddd1793a1b3dba_567_472_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\44716b0bb02f6eb446c86279138e6e4d_222_4_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\449a9b8d6bfa40577db0f7fa9330a83a_426_4_14.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\44a07a89df8a486f1e90a31c5bc37f2f_294_1664_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\450bf30d91921b54eaf9ba2c8bf93897_756_1948_486.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4561e611bca7e7be65b96550a790d9df_552_1948_736.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\45adf6c40d2f9f688e5b38a0958f14f6_179_1974_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\45bea0b13de46330bd3bc60c52ff756d_394_4_134.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\45ca782cc62d42aa67b6250d6bd02bf3_238_4_286.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4625135cfd8584d84d51064533a827cb_336_0_5.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\463c00b38c2ebc9057e8235ccb715bf3_623_2_1013.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\46c100b7709160f56d7c466375be2dca_435_1414_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\47247de9fc903f28dc72292a153c706a_592_4_762.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\473841902fc979c01ec90d7349cb2007_479_1098_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\474a19e9f163150d03b7c3834fbf22ab_34_856_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\476f5f69959e40cf1c588a5091e770e8_288_1110_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\47d541abe486b956c24aa5292d70d643_516_4_510.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4805eef809a89f3db75f3e69f4c36dba_765_0_1783.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4813605c9e6560179cd2be78818237e5_723_112_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\485ab514b84c4cbffa642d6a18e4f45d_155_432_1975.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\488773d919383b02dd60bc58fac0b52c_94_4_1708.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\488b93e1818f2482ee6b6a0bc0474e3b_514_4_130.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\48a7fabc48ae9ef2f697adc21e1bdff8_710_4_96.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\48dac1cc525b8028304cb76c3c46ffa3_658_1480_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\48f0156770f117c154b40e9d9a4f3f67_222_4_1044.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\494de9553e0f8aed426e124321d24eef_245_910_882.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\499491e319e7649c0a97b93fa5d0580f_411_4_502.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\499c69274ea3c48a70570396e5958dd3_346_4_1062.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\49bfd587a7567515720cf3730d0dc65d_732_1078_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\49e36c367b7337eb2a0bbb78cbbb92ad_221_1998_1888.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4a1c69565303efd5be0251a11cb2fecf_497_1952_854.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4a2b6b9284e603acaef954d049de7a4a_9_1988_1438.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4a4175acd1de636975b0619bde673f34_184_1432_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4a4edb75cf48645de29c5a0c6c451c1e_285_934_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4a5170040af537804e4bd28db82ca106_276_2000_1393.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4a6b1b6fd5b355f8ec123cd0e01fede0_202_1976_764.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4a923484983865b08749df49b493dbdd_102_1976_1346.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4adc2c1a7b699cc66cefc1a631f08b5e_164_1940_1444.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4b032263bd2487ddca1756e59ef18ce7_94_0_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4b175a589c9caa183bdf2907c37c5400_516_4_162.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4b2660b7363d01c45017d301cff5b8e5_669_1890_1368.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4b575263830d20caad4428a83fdfe31d_424_4_1216.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4b8786d1de354a9c3623f808e5aabf46_405_4_526.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4b925878218768157294847485768539_477_938_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4b933e435b4e5477185eab994320e437_731_1574_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4ba99581f5d068ae3d2740e43dac7b72_214_1928_988.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4bb9b1673590febe0a3f976e907caa6a_307_1102_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4bc2a6fd5b8c9e23e34edd965ed3d8fe_276_4_136.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4bd564a80b6a920fcf90e3ecde5b6ae1_559_1670_58.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4bf7e05796f12008e3ed9479413425bd_1_1858_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4c193c1b9a8259e39434e25ce6e63f6b_426_284_740.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4c59ffd38bcf93bc65b5b50a3fae0d9f_505_682_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4c7ad1bce482db42f393c2ff851f197a_210_528_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4c922a599cab753bfdba00fa8001c8ae_571_4_1448.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4c92764ca938d8ce477d86d47f49a8d1_714_144_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4cd3af5d95467427d6dfbc967f14e996_488_1290_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4d9192a98b93395f3bb11b1ff2797dfa_112_0_410.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4da04793383c84388c379589c560ea97_119_1966_1134.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4dbbc72d8e87138b6c1ede1ec6b185b8_10_4_270.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4dd921c2c5bdaf15e770c7654e149023_103_986_1602.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4dd966ce534fc9f9d8dba7b9df9ea879_208_1020_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4dda2038029016259238cabb6a3fd991_31_4_1148.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4de84a80065462b9a875079ff5cc0fb5_190_1992_756.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4e1f4a3e59608df14cb3ae5a0934e155_32_740_248.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4e77454912dee75af335e729261b4eaa_191_4_1542.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4e8323576da558721826be12f62943e1_375_1488_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4ecc28b7f8e58d33f37ebd01b676f677_605_1948_796.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4ef0e07c21c5bc1730ae0fe3282012da_312_904_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4f4198c5944d75aa1935ef53667d87c2_608_1006_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4f9f15d836516479101c6d3ebdc85098_216_278_1570.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4fb8dfeecbc61aa012403be1adec34d1_385_4_1332.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\4fffcf124d1a447e5afd10f84931780a_257_4_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\507528f79e4780489824492e47cb2c4d_49_1980_1222.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\507f579214613551ab239d72df140e3c_376_4_1330.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\508448398ea8db2d92772ca1a87cadf7_272_80_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\50960726ca374fa6a74560a63c3b5cba_91_4_1456.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\514e2c5a2fa8fe1bc5d5217f972b2837_377_4_48.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5165f9effb6d5170f72f059b6ca83517_480_1208_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\51961e669e133899f122b98e62d17d35_30_1682_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\51efd413e0ac6568688412edd7ee1bba_401_2000_1826.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\520a9dc9feb0a5ddf787f3484345dad3_135_1024_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\520ccf43a0cf63a8310b22876c2187f2_454_1196_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\520df87f70fff66b3d47fd98d6f89065_299_4_1344.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\521e9a8e96a47bfc4a0b0582b9bb7478_694_1890_1154.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\528069164015f738b091a93d0333bea3_590_4_694.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\52a1904e33467965b7a8f75cbe65b957_491_4_860.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\52c035e124aae0c8432145596996d189_151_2004_1311.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\52c4ef114e562999c1511d8b23f29392_683_1962_1224.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\536b5da1c7d1b1acfee4c55ce816f717_312_586_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5388fd106047ab9bbc37e98272edb16c_57_1956_1488.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\53b8cb70b4ad7b3a12cace7183e5c143_231_4_1584.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\541c1fb7e30862bb6a28b2775c06cc49_319_1522_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\543c6b0327d269eb589558a16d2437d9_330_1472_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\543d73e0416313a009f54150a60917f5_322_2004_995.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\546e617ea79dd3b42f98445ce83a8c52_155_0_1120.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\547467da9196f60e70d8baf10a9725ea_556_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5475412c9128185dfb0fcab17174870d_508_4_344.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5495d8e529bb431f1498e551698d15c9_481_786_948.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\557825d76e1350e2687881b895e3d129_418_2000_1999.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\55d2b9668308598402efcd1065d7cc5d_432_4_896.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\55fceee77523337f474b514f2851ee50_539_4_1728.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\565bb272b99cbd8bf98774fd478fc5e8_197_1072_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\56bd4c8c655b9f1e89b1960ce510bebe_522_4_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\56df518c11ce848e3211de269a304adc_448_1972_1514.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\56f9f8263349fa8ec6bf352274a466f9_455_0_494.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5707e38244a2c0545e2b556eda489562_637_1952_87.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\574d374a71331953c50908a42e11bdd8_511_1490_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\577da4db73c80b581aa1a21737c9ff98_457_1426_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5785f494583b48a47cfb6a5125e4cc72_516_1992_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\57e22ed61c7e97233e4a023be13c926e_416_258_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\57e45f423766aa25a88279114fa15b3a_304_1996_1646.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\582ed8070985d27bccc6f5f8c2541f06_424_1990_706.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\585874dd951b2d762cc1330deab6c195_683_1732_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5860c6533c8e8790c3ae36952485c20e_69_4_422.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\58cdc324488ecf17b04f0b397a77e2b3_793_1978_1294.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\58fa3a2f7a98606a5c300648cfd856a3_422_0_1436.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\590fe9f58ae43a9ceb5ff6c8db8555e9_483_1932_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5934273465839c7f91592c5c21fd4506_518_1934_752.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\596341f113c9d4826f256d4ccfefe2d6_335_0_1247.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\59a3a0547fa27c9f8c28a7ca33d499f5_146_1408_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\59bf230b53195510b682c13ba391a5ff_723_1980_1310.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\59dc863c16b0fbe0b07d1987870c43b8_565_32_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\59dd305fb7d63fcc59073741c2635a18_688_974_320.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\59f62fc09b15bf564a43bcc36604f782_511_1978_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5a5f06f1335a6d2ba943717cb1475e28_789_1646_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5aa5101cc558e0572bb3cf6d75d53c16_503_392_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5ae6bfe8fcc4e888b343a156f4085015_367_1198_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5b215c6fc5516adeac2117dab6fbfde0_294_528_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5b809f2c8d8e42482ae013283d0e5b2e_211_774_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5b935ab6557d26d7990d9658e41794ce_458_1946_770.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5bdda095c45d7ceabcc253f2aab239f9_192_926_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5c37d9bba8f613679ae6a53b09b0fd85_418_4_1534.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5c8ab0205ab966d75f005b06fa42723f_421_1880_372.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5cb27e8d7350fcbcebed34b130c6989c_636_594_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5cb346bf952b8945b23d3f37ae3cb787_323_4_1498.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5cc8c121c6f4a7009fff24e635340c4d_127_4_94.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5cf8bca39b06bcb519d0e026e8cd820d_301_1044_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5d26b1bf9bc50af57c687997322ad9ff_584_858_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5d3b2bd50f364fa901797fbac18fa814_29_1218_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5d97fc8b1a8b90b778f3d32a31370e9a_210_1984_1556.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5dac7ae477a4ddff93e13342a1ecf9cd_184_216_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5db35fe7c1670020c0b44347783ae2a9_119_1984_1492.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5de7ce9a254e8cbbb5146821a6884cd6_143_480_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5def9edcc5fce6fb69d2630f3a1f315b_107_4_96.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5dfb8c835e3949341b2e81190a3ee4a8_202_4_474.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5e014dc6369f33dd0229661d95b87678_30_1984_584.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5ecae5ace454e0f3eb5320c204f3fc8d_566_0_1489.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5f1ee5f9aeca2c2acb91f53154f84743_546_4_1216.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5f6120177358f15fd099d51ac49b646c_297_1956_1369.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5f7ed428364bc4c171540ff8add7d113_640_1360_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\5f8045f5287fd9f824d8b22063b31a4e_50_590_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\600ac034f2e1d85ad5cd6eb24855d2d4_650_1882_264.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\608a67f1e23babaebdc08a0fedf13795_202_600_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\60ab721ac8bdaa4b4d36e399ec973d8a_418_1854_1231.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\60acc2b3c66e8107ead1adc562c89f02_508_1568_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6107f90a435d1e81d38b2f18b515c4bb_563_1364_902.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\614ddf9d425834b2b29a1a03e1968850_477_1978_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\614f226eafc414aa2ee136db702b57ec_102_4_112.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\61784bf55e514ae6d65f71f7179320f8_390_4_1822.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\61a9d8d5219e303eebd8141592c53bc1_80_1966_972.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6225046d003aa94271f2fa6573672fdc_620_838_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6234c2d4671b02e64b607d161a7b9b74_389_1458_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\629c7750281a9e6b9d6aec90dafa7520_545_430_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\62b73aedc2b2c85c7aec39a0b63ee543_305_1994_374.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\62b8362c6b76bc0e2474e45415c6d371_293_890_103.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\62fd9b1096a798794826f6e4baac2c87_358_1016_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\62ff63c1e62bb4f042c4811b00f36ee9_185_4_1130.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\631d1bdedfe54a215ec70e04ba00e573_212_1988_400.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6370f50d080f2e5299ab3c192bad4ffe_644_408_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\63f4c63518f9ab8ab26e292f11c7512d_220_4_1122.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\64bdb3b3cd94fccab42a5904f49c37dd_189_1388_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\651d237b5ae85b59b53caa09068bcd98_583_526_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6584b8dda41fcefad62340dff3b376a0_205_1918_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\65ba51f6a97daeb0fa5ab26e17b55eb0_144_0_937.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\65d4fd2984f5496d78116c95ff577cb3_88_4_918.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\65dde945dff0eb08ea79b2f0acbcc5eb_551_4_1594.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\66525c430c608385e38ef244ab2e24f4_664_1666_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\667da9c6a951b782b7d5bae0513dec83_502_4_1040.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\668872b18a5b01c0615242baef2d02fe_54_1438_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\66dd044a461102a51fb56f3bf8711a4f_677_1320_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\66e395ef8b5a118345758d77a8fc30f0_500_494_1860.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\67498b6cfbb04ce87c63629f56695c0f_440_1652_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\674a1431d3c06a1f7655fde2de92a53b_663_1868_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\675d7cfb3bb264e148e4a0ea8ec253ac_204_496_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\677a84abb5e0f1f51f69f7ec14a0cf3a_219_364_916.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\681113c2712e1a40a01bc2f5ce746748_401_4_320.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\681521c3e900adccdbf7c5d37cbf4362_100_1892_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\684e415e39b8bb711b909e9eab270c68_313_702_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\68ab54fee37bb3f7d456cebd2fa1ad0d_501_1990_1450.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\68cb7bb2d801b70d97ef16f56cd898d0_568_1462_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\68d1150a7267ac03e6264bb731bd8447_717_1984_1256.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\68d5593e9af818d5eee007a586ccfd01_622_1480_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\69ab8a501b3a5e912c525123d765a2b0_42_1972_1514.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\69bc7f8e34284318913782452d5d3758_516_1610_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\69bf9bba491aac3137df1019783baa88_778_1452_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6a605643c2ea5afe0a91f60740ca8b03_617_1956_1279.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6ab6ed80842a2400880c88dbafd20e69_282_750_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6b2763e8b99c20a2194c42da9a9d019c_258_4_1460.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6b3bee21b680d512f9bee8add47a3183_307_1094_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6b62554a5a7e4e70be7c1a21e8b0028d_531_1904_1274.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6b6ed51f689d196a0e1adb3b285124a7_18_4_324.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6b76021ee17eff01cc58f51eb52959c6_584_724_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6bd6c75c06666a4bbe122a91bdb99454_267_128_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6c14e5dbaabeac48fc28f78c6fb45d77_136_1986_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6c6e6ff0cdbe7631805cddab9d0abf81_530_1956_1546.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6c9699d99292660b6f8cf13f579ff848_186_804_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6cdfe6d1393fc0b4e12b4b6d7af3c9f3_618_1600_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6ce22aed67d6962d8475d75d2018c75f_140_4_260.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6cf3a6643e89cc89ead8257c3aec9ba4_388_1234_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6dfeadf529c54729a3303936b03eb6a7_759_472_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6e4d1261e33835e713137f452caa3046_783_1990_626.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6e59e386309a577309a5925f0b74bc2b_29_92_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6e7c177e7ebcc7b91ff6e4ce77f88f35_120_1964_760.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6e985666b8925501f4d00e45cf1d05b2_483_4_1788.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6f9f0d544a5345e8cb46e3dcdbee8738_570_1000_1672.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6fa4d467c55fc10794dd5d64fc8c7d38_180_0_1832.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6fc7f1ede4c8be8a22ea938a87bca2a4_24_1206_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\6fd2640b5ec012d662de3d38ba3f22c4_154_0_1691.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7007a180eae32964e6dc2640534f4d33_166_326_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\702dbadb5dd16835c0cae6e829c4b4cd_108_844_199.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7070e9a075e3b3bfcdcb9e299cc8a9ac_561_1418_1971.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\707cd447544ae69e8b01688439b302b7_178_2012_1371.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\70ca9ea3db3f7294c5062368f121fafc_567_1042_1371.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\70d6733aed8afb16e78462d34b356605_387_750_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\70e186081c6412b47bdf41d4af5e59f6_194_1332_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\70fac8b43e366e36ee84031f2944b429_747_1778_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\71017f3ed18a120d08f85b92398deb94_285_1848_486.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\713aa16e3e337cab9a72a55bc1e45933_427_4_386.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\71aa89cef110816d38486e639069ba8e_404_1704_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\71b20aeb02ec726bea424bc83d983bf6_404_1988_1220.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\71c3fcbaf86ad5a7207a440e7e052f9c_282_526_1264.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7252c55d4fed0d618859c7821139b2ee_423_796_466.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7258fff925ada0a3f6f8f710e7832ccf_290_1046_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\72a9650942c47ae5f8741ab55982b445_493_4_80.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\72dec1db50fbd9f81c4244a081f2b496_734_684_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\72ec85bc985b13bd128be2af3df618bb_89_502_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\730855f2ef2439e00791970848557766_35_0_469.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\730bea7df1dace52efccab1bfea1b499_552_0_511.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\731292815ae5affddf1a4a7564769210_289_4_1568.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\73fe191b651af036016724a67a8c2d2a_193_1364_240.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7441963b5e6b27c13a2e207f6091faac_651_4_764.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\75292fb725588798af08bf032ee11fc6_86_2002_1728.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\75295e70f269e5477fb865517edbc6ae_625_474_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\75b76bf5d06bdf10460f6f603b4e636d_549_1526_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\75ce128a193683a5ff2681a535d92f6d_437_1258_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\75e3a6ba3bf8a69ac0c0107b27116450_317_4_1066.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\76185e149269e3b524c4032fd3f2d352_15_4_576.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\765ede1a514b26079d40f837fb625285_126_978_805.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\76763b747ed03b8857515832057736e7_220_4_1158.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\76a9c0626d2c302552f74759132484f1_399_932_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\776d191b14f0bd78520ac4b8e33dde71_104_1980_1204.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\77711bf12b658ef03c57e259103a96a3_336_1232_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\778398e225c4d666a33d871e67d573d2_746_1488_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\778e60944784b93a732c3b9d92299001_696_4_710.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\778f3f2cddb4f9629492a346f58b55c8_494_1908_858.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\779a9c0bc86465993984d10d509a2409_18_46_1011.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\77cfc7fbf069576af4cc6ab154cd1c0f_104_1950_500.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\77e0d6c66cf79d1d25a5b016103c0783_769_4_62.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\77ff8612243fbbacf15d52b296650e49_222_1998_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\785244ab8622cf9d52d11d2dbf8dafee_493_4_450.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\78b8603155eb77975ced28576087396c_7_1914_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\79715f6b0e83a6e7989e2a3d7fb14bb2_620_4_1752.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7997d155dd0bfb2a5ea9f7655a35acaf_623_1990_1692.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7a1dae910a2d75e305cd0e7b97822401_636_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7a2d5b6fd3478c82d6c3af43d92c79ca_455_0_1425.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7a81bab1eb4a326968ca20037684f59a_47_4_1416.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7a83981f09a4a48411260a6593fcd425_525_52_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7ab0a02158326fb3756d2a03786ae9a8_175_1892_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7ad76c74a18bc719218736fa58ef9938_523_1164_6.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7b4e20b42218eb59194fbb1f0799b255_479_1660_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7b5c3fce4b74cf22e4d48f2e561b96d0_340_4_1724.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7b9e0259e1ce8b9ceebab45cca640ae4_519_2_994.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7ba5304263abd5e7c89b5f7f3fa4d34a_625_1080_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7bd9d5cbae998ca43d8ceaa44acb80b5_614_230_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7cde03fb54f2231511e14dee70beb96a_638_318_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7d14993b91776ccec2af443f700b3436_248_1956_1350.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7d433b8a03dfbaac519270ff76a8fffc_9_4_24.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7d7d2f08cf08e38f4a3f0de9098aac5e_140_764_863.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7d94854bbce045e842fb74982e5a7f21_297_308_1270.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7dbbe16c545da8dcef74a1529bf4114d_343_1996_754.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7decc086c52972d0e22b87d138c45ae3_121_1908_1692.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7df075b37923e280009eb8e6ebf4a941_178_1170_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7e0a7e7461e0470661b53771cdb514d1_494_4_1210.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7e250002a4f296e533a6bb7ffb68f081_520_246_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7e6bfe66e5d13aaea1ee0c6bfeecca33_180_1464_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7e8fc5c7d9a984e2b4cca7530ed802dc_220_1958_1512.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7ea7822989addf4921bb4d141b1642c8_490_690_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7eb535d7b0b358b993ab2bb15ef530cf_32_4_304.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7eb8ba94a66cce18bb8ae765ef64b16f_272_180_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7f3cab49a688ac9910ae91ce33a93e85_465_1978_564.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7f4abfb9ff703a1afed584b9f1cb0469_437_978_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7f596d80e4da770fe27939b6ca486c47_432_0_1182.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7f77e23d61568a5395b83be548d01c4f_292_4_1180.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7f8ab852957f63d541bc70faa1f87acc_728_780_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\7fa02f0ea1f860b65fbafe8f1ce23adb_356_1784_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8018fd987de2009406bd0166f9f42722_47_1952_1030.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\802c3f3eb0dc6aa87df6a1ad742f342a_710_1132_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\80ac7019a73e8984ccc69c433f47f759_512_4_1838.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\80d2a77bf762052db057eecb0ffdac18_664_624_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8160e80d8bd2d786bbe5d5e3f29e7960_34_1948_1354.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\816efc06c33d0bae8bd192e6174cf3bf_49_1476_1176.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\817b49337568559d0a7b7fb0802475ce_302_1994_590.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8195c0fc3a7072379b9fa32e204c2b9b_317_4_1082.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\81fe8583e6a2bb13d634e22ff85881b0_447_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\82708def14bdc3e76a83ed3fd3abab44_34_4_1134.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\827dd01a2d638e7867492d68417b6ea6_533_910_390.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\829207a6dc3f542076ec7f4784bbe9c2_570_262_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8295db3eb0a06ba63ade28375594f5af_526_1590_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8296967acc334cabd275bb97a950f6ef_118_1956_1736.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\83024f923795445ba107ce005aa49de4_233_152_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8317f11d7c96c141cbba6b249fb5f4eb_780_942_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\833915c34e0e3c5f13c7d1e0a1c75a4f_514_182_436.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\83397d30fcf4db331f8418b32b97b0b2_319_572_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\838a0a1827fb733c2e0c39f5720b894e_72_1750_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\83a667e0a9cd6c5b9dad0e48043f931b_371_4_402.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\83e1446602d3de93b784bcddc293b7c7_602_0_878.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\83e82367e89d28847969187f008e7371_367_4_896.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\849ec2f26fb4b30cd56ab6aeff656c5a_329_938_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\84ccd77b38d73407536af89237e804b8_134_20_566.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\84db9a12408e1a0521f77284767008d7_31_1612_908.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\856a0f10f01c18a6880ae21fffb3986b_669_4_4.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\85976cf2f16e3d3f1317c2906b71ac77_531_772_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\85cf0177c74a347f7ac43e947e54eeb3_32_1952_1815.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\85d0a39489b73b28242647b55a38aa8c_785_4_1662.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8618b778a692ef880a167583db783665_222_942_1770.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\863c492045a18017c61e31667514f627_522_854_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\867cc4c26e5ef69179c7cdcda42f6bea_735_1106_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\86a3f93b9bbbb340d6bd91da52fe686e_647_4_1746.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\86f57bf82d135dd7e624c4cbad918903_265_1606_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\872a760eec0bc06d6e27b7fd06fdfe96_552_368_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\87449fc30f066a44c7585f85739fe302_477_0_1675.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8773972f6ece08c87fcd00d00e15fc4f_712_4_1024.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\881293088270af92c5612e24ee1309f9_289_4_1170.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8814efe48d6d7cfc367d5a1af1e04416_75_188_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8829598b03a88538aa5f5e7f30e5fa9e_224_4_1826.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8874660c6666adb78fd5f210831ea1c2_430_1384_371.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\88d3d1d39cf765b4361395c1ddf5794d_213_4_766.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\88e328f997fb0ae318868acc8b8ac6cf_519_1636_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\894ab46190a1a5242b6025c6b05cfc9c_180_820_562.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\894c95f90eb68b683d21fcd614701daa_381_866_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\895d4e94d986b8f87397beaeb6c71f84_702_4_1198.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\89e664cc134b1bf02dfd23e2bfb7a919_548_38_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8a24f2ec472c285132326d39a2324511_374_1600_849.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8a6ae8b6bd22c4450283f09ec6937fb1_366_258_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8a9e871888ab16f20c0ffc14d586f847_527_150_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8aa1afda96423b6c2e0c5058e6b5e3d8_625_4_1172.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8b2c0b477cd47f490ac8c2f609556618_699_418_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8b5c3adc65f87c153e11048f033c43f1_219_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8ba5ba60b4d344bd625c802afbbf0a5a_488_1992_291.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8bcfb20cb50ae2091e4c1062ae8547ca_795_1348_1164.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8bf2292661cb875dc4d2716dc35d7102_621_866_1573.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8bff79d02256e76cb0cd19b8cf108aa4_528_4_956.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8c3b950e895a78281de13100885f8d8b_209_276_1110.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8c6460b834c491a1d5e8aebe43a4327b_567_4_698.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8cc773d06abab720a7aa65a0d841b8e4_175_182_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8ce823410986c1e664d349fe63f60f4e_226_4_1022.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8ce961d30ee32b8dbda46e4a17419f86_85_608_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8cf9e224dc1c7d841863425a6a736a82_368_4_760.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8d24d2b3130d2e7af15d1bc982259aaf_657_0_1443.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8d4875a2296be454da0d9acd2e07c1b6_427_12_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8d969d73c5b0b062d3a693d74f74c840_454_1972_1086.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8dad4af2438ee4de3335a9b148f96219_768_4_670.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8e283ad97a96aae33894260ab21431cb_391_848_1276.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8e31eff68274ad4e7fff33e4b54cb0f9_322_592_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8e36b793573f2a5a62e9e8ba80554a27_479_1956_813.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8e8e633d5894663ffae831869acd6415_641_1338_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8ea7c5c99cce6419e08ef08921983a80_214_1410_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8ef4a1fc626471e9a15f257fe0c5a9b7_555_1292_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8f1564493e10daedac00712b1463c248_256_1212_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8f3ed810065d7e541e01ad175ef22b7b_434_4_1628.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8f87813dc265b4ed66194e6006807fec_659_4_98.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8f8e34a14e2884f87d921ff064378392_83_0_679.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\8fc0bce67103fdbf9864c0fc66125162_257_602_1059.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\901b7324b18285eda9855767c3736d89_510_1896_1336.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\902ca1236278dc2d8440d2b47a965f40_452_554_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\902d2d38e1367423d34158e3fba5df00_550_1968_146.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\903f06c2936879f526aec5150b19d890_400_1964_922.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\906e9f0f8045b7f8da6505d6a4251ba2_104_96_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\90bb5371a33b840f943b9cd5aa57e570_617_4_1148.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\90e4f52c290beb644a4dce5ba2ca6cdb_361_716_1452.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\90f33a63b95ff8708eb61a08395d5946_577_4_1540.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\91712ac3b2a810c00ee19d595b086e13_631_416_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\91948032d805eacb36f6c8a40f8db966_541_4_1630.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\91c5febd3084d7c597ad4146b552fbe3_147_4_990.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\91ce8f933c9e5183bd54e4085076f06b_465_496_1332.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9220802ae9f127b896c9439ae5ed6ab5_572_4_1784.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\924e853a576fb4994a02d27aa014e67b_24_1996_658.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\927a13135d41f09ba6547d809771552d_599_598_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9294d329ceb51f515ff30f7532af839b_430_394_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\929f2aa21e4a80ad74b48f614fad3fa9_475_1734_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\92d81a6810a3885beb70c75829c81591_19_640_914.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\933ffac465ec28778ee00744ae949e74_323_4_1955.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\93fb1892e2c02d7dd048c822cb736203_25_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\940d2433836d907bbc54e9cdc284c5b8_305_1774_292.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9429cd03997b4b1209511af24ccc39ee_432_132_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\945d0645cca88cbed554e54a2f06dc67_464_4_1070.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9473ca341e0fb6fb97cf208cd25e866c_499_1070_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\94b71f8eb96dfcc007975e70af9bf4b3_375_4_713.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\94fadac8675eea5d4292851e6ab13432_266_616_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\95830697e7cb63338d5852e320c472ca_213_642_1858.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\958588c9eac2c7c69cc09e3b3f8de6ad_613_4_1090.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\95b957fed25575dc479f57b38d990958_622_4_1666.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9636fd86a41c7d4f14aa34ed0894d2c5_370_1362_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9657a3d2ff5b8874d68d31403c7db2ed_652_838_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\96716d1e312d14ac92690fe63e2fa7e9_699_1892_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\969a0bc3e1a0fcc819a4f3519140be7b_610_4_1768.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\96aefe9c384fe97b55fd2ea088a726fe_30_1968_806.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\96effbf9af26d756253f92c3596d554b_19_1332_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\970e7d0e5b76de33307ae5f3c69506fd_271_618_686.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\982c5f73f1e2c6bcf51b54efc0281163_685_0_1662.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\988d110afafb66738354b5e38edaf01c_454_348_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\98f3a8d09ef357ae395e318206807c14_316_1968_494.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\992c94bc5c0cd6a83cb7221624d09538_460_1952_519.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\99871dae2e0925a7618cc801701860f5_376_1374_1786.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\999646bea4f68af803d11e5cf4f846da_214_1532_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\99e8b95d7bb2afb282757de16f8bf74f_48_1856_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9a3b8310ff28c36c3b81c8554f9d5d35_459_526_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9a5fd297221a837d8e47e1309823e78d_330_1278_532.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9a7c8dfd898aff2d79deb9011c27d8d3_604_348_1171.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9a99000a47c945c0738c64c7c83121c0_252_1334_792.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9aeb5f8fe926e099fbcadc20c7f60e84_517_30_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9b34b26c30e9e50df2617a0526cfa8e8_430_1674_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9b58b0ac8bbbf83cd5e105959d7da1c3_477_182_1761.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9b5e257b6037ca2fceb17a60d6bd30e4_528_732_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9b7822330f30f0ce08c659fa2d8974c4_352_2004_1997.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9b8130ab439ba8e21ed8451c7ad82bf2_180_934_1290.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9bcbb495e0d1186daa129599a46dca6b_5_1130_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9c4e2ff58a3ae3ddf24a5a1981933b66_90_398_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9c55e115166305c260afc5bad0faea76_442_418_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9c6f4f9b2a26080d2545fd3914906bc0_697_0_1437.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9c9f8435f217811bc72ec9c70d6d1235_524_4_1874.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9d539dcf1c56627498c3b4b542a7aeaa_433_4_244.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9d6f69b64e410f92b5781f07208da464_395_1948_900.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9daeaeaaf0aa5c6ef35bb0fd70fa089c_612_1980_609.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9dc26a42c3aacc25b76b6eb651106222_171_1980_914.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9e113b8c43ce774d1b23cb5e2a26dec3_541_122_120.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9e1b7673ac19f27fd5ec4b2eb309528a_333_1984_1983.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9e3f672e5da0ca3db6c055fda1734b0d_584_4_1612.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9e691aaa348179ee237f778a2a2c6c35_704_424_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9eb4a6d07f09053e6f8cf27142ec74be_527_4_644.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9ed20d1e158fdd541ee50c425006b448_621_1274_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9f22a955c4189f232e5c6144839ba38a_509_1358_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9f4c2e9e8b1a035d630dd9eee0bd69e1_285_1964_1037.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9f58038bbcbc4af39630756f824ef306_488_396_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9f879d92232f76699144850fb8616531_407_158_1646.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9fe10507e8f1c48b5f76e5263d150f26_298_2008_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\9feec265d2ecf82fa15b1afe184465a8_111_4_1110.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a05d9264867da77d39ea368bc7a38127_263_1630_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a060c83f3afd895372d837106048f98a_186_1624_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a06ee2545da78aa0b347e195ca167741_296_756_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a161eda61e545738baad73b28f1b90ee_192_1976_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a17a53f641c2e5bf1d93e49b6c81065d_138_1964_1129.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a180dd8520bf59fd217205c5ce679b49_109_4_434.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a1a82dacac833121f2c04c84bd83ebc2_366_1914_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a1d13ce1a7d872d8eb2cb810c1a0dd17_485_1956_585.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a209657678f9d347388ea48bb30f2480_320_4_1622.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a229ddba70752e69b8c498156397f156_258_4_150.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a2bff78e5aad06d24950ed1eab0aec41_203_4_1644.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a2ce3a224e031a8fd40b4c265a6bf586_413_1098_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a2d88fd038ff0f0306c7bf419509d873_483_804_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a2e4d9c5f712587f04e53fe5bd84161d_334_0_365.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a2f9a1495f352d073a7b9066f0d64f66_244_1468_444.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a324433a48b37a727aaf66cfed6d5a59_151_0_697.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a35acb70d001a50dd15fd1a6067a2f83_692_2000_1750.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a3a297460e589ffcaf84c703be8b0fe2_83_1286_1533.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a3aa673cb3c3de985410994823ccbfb6_133_1100_726.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a3ce732f206864fbf3ae52c1947097ef_264_1026_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a3e1af4212db493034140fab5cb85ffc_305_1964_943.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a444e0eec046401a0948eb9446b6d40d_249_1952_1563.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a44a583dcf2e8b8a416b527dedc1cb1d_250_4_746.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a44f2f73edd03b94a91f8e615ceb35b8_576_624_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a525746feddd5b5368c1b7dcff3bff0a_310_468_1631.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a5568efde72cb0fa97f0f5a511fdaef1_277_560_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a56bdfee5018bb0fce1657ea9b055ca6_605_1976_1818.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a58e357a18ae748b715118347ac294d3_34_1754_1416.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a59f9dc8360a9ad77a96d3ef1915d241_107_1988_1462.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a5a8e78c267cacc9ca759bddd2c044cb_586_0_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a5abc3b401ba8ede283a50ebc335b205_408_0_1707.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a5b06950203f7952cd321fda9bdd5b59_291_1952_761.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a5ca33725a53e00bd252a5333ea2783b_73_546_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a5da8ab02a1674a72ecbe507b4f08acd_468_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a5e798c4dc35722f36671778500fbca9_79_4_170.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a6009890b2dcef2ce3e4a58fb85ca52f_575_230_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a648b94f33680b395146c40fc06736e2_38_470_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a660fb2e06b5e348ac1b89bfbd68fe2b_466_1124_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a70e1e8102cc44e70f44bcec928eaa46_265_0_1527.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a72ea5acf1f97521efeeab8ec6de9b08_319_1348_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a7304cab951d86df7803bdfe4aa033d5_264_550_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a76dcad4395d8d6de227052a1aac66ea_545_4_1326.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a775fd68c92ee9e742e2d10d458532cd_22_4_1436.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a77917d11f916237d7b72c05924caa1c_719_1152_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a79088469be6cfe3235dfcea3c84bd2e_72_2012_2011.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a7b93a1fd0142599f1b34f6887b0c204_391_4_714.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a805ee871fe71d4013e4721ed6b49d7f_46_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a8340ebdb4e7e4350b9af6d45e0adfc5_697_0_58.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a835b5d886ecad103bbb9890ef8e8f4a_259_0_1615.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a86de2c32b1dabded4f64e8e051b1fab_579_1556_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a882fa57f698ff4d621c351381dbd0fb_393_1952_775.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a8894e9769f0cf44549f158cf71ea2a2_738_1576_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a8b46e3e403ad9f4b614e5736fb5ea8d_496_1972_490.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a8bb653c06f56b9b1dc7e5a3145dccea_673_1070_1162.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a8c25960fc7f143f03658b5f585161a5_372_1960_1447.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a95892ce21700e817a55f7cd01b37582_524_674_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a9771bf9433572238730df647c911d07_280_4_426.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a9b414f22e4b29a69d56fe45a2f11c00_323_1932_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a9cdfbadbaa3d49d6f23eb7e96fd3b8a_59_1960_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a9cfe1c1b6e08c40690836554f8d99de_77_724_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a9d025036bc27747c375f18a21fa19c5_188_1924_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\a9eff0bd48ef135ffba345f312ec5564_672_738_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\aa11c00c3ff93adefa0430acc016a527_397_512_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\aa270feeda2dd742b33a6fa238e99ecd_724_1194_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\abeee6773c45ccbce524626d84f7b3e3_479_348_695.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ac1b9ac7bf6b8190d9194f062cc307dc_342_1338_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ac24e006b9bbe73f8890dd2ba4bba9e9_103_290_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ac95a62c8aab0978844b18b0c6456ea0_105_1982_382.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\adf5f55a8a94ebb18dffa05a46ad30a6_702_1980_762.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ae0f18364908aaff892cf910bf390a79_219_1364_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ae34af3eb45f4df52b6159e1c94a2b9c_86_4_1698.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\aea7676048614233e1b58c153fca6548_724_4_254.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\aede1309614dc91cb4d621483ebe912e_581_370_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\aee41b7c0a480128789cdb39f5f054fa_75_4_434.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\af3cbb29233f50b3e2e662f7afa078b6_445_4_814.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\af8774d7f4f909fd2e2a557930961417_372_1952_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\afc6568cc3a822955eb60746bec12d93_341_0_1964.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\afca3a92add15dae79cbf4a803d5807f_40_562_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\affbf46c81093f8d0c399e4a6a123391_486_580_15.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b01d9d8eadac34c2fabff86a4e992f8b_270_1010_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b033c242f00e6c355c2ad71c727d6f1f_478_1488_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b05d1568da271c4f1f5aac9806371172_101_1594_1293.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b1304356ea67ce7336d98ac553efa8f9_223_1972_535.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b133d0c1b5023aaedfe2a3ad368dd307_576_344_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b15fa16df94ab91f607787fa414ce32d_547_806_1050.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b1d8e317bf903c89b9f686308230f5d0_351_1172_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b21a2b4dc917bde76e984fe3e6c2cfc2_600_766_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b21f0fb8ffc6fe4c0daefb59adb47ebf_350_0_827.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b2333a7cd1ae2388eb7dbe4ce46f94f5_791_486_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b243805417f25881fc6ac1b3f4fcd364_340_1962_338.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b2b03a76cf108a7cf03be0944f238557_363_4_142.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b2bdce44d67a7f5906a37b062110fb52_640_0_1465.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b3431370950801f71b059bb9acad8141_351_1600_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b4032ba392461b0da53f142f428556a2_125_4_1564.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b42a781288da635e3bd6eeec554a1aad_362_1996_1227.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b4493872cfb1787338cc59d5189df234_591_4_794.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b45962ac41aba30df4c65d354370d457_649_388_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b4813705ac0fe8ec4c50011c22a50a10_27_228_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b6013bfacd91617f6e981076a97f6a14_204_132_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b6070352f16aae380c0e33d10cc1b5ce_109_804_844.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b66e301bb589e2c99324a5c565ac33f8_37_4_1568.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b6c0f89f8e1a463ce18c8b9654875ec8_332_1952_1029.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b6ecf8e10604da3ba6d3d77b7e2316cf_449_468_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b7085e58ba373252187bf980f5c471bb_606_4_896.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b73b42238cdf4f8372cf4bb7a34bff77_553_1350_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b7671f86291406cd96114b4cfb3a085c_292_532_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b77d69e4596d9d74e90a41ffc03326f9_135_4_712.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b7a5536f994b5a4cbd6a048ea19cf5d5_477_1986_40.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b7c51f7f1908d4bac2b0adf0244f8730_431_302_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b7ee9dc64f73324cd9af09691af19ee4_225_4_1024.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b8892dcd53124d7f3e9a62f6beb2e33f_277_392_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b8dbda23164f5e98ebe0911f428a1ed7_354_1992_986.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b93082fd751cb1a049cc7bdb798a19f5_30_872_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b956ad03721bd987bd4daa0b08aafc51_704_1332_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b973b3c05f9212a917020c098314f0e9_113_4_668.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b9bf7a1a956fa7c36de174638585c816_729_146_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\b9d358c9f7ff8e3dcde53ad8132f9234_213_4_1656.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ba15cfcb74fe37c38b0461109aae181f_474_408_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ba52ed96b8c65b265b3801fa92757874_306_1800_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ba7ab8880707ed0b39fa34de2d6fa452_710_32_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bab38f9a9d91cfdf9277051e945ca510_277_1466_1959.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\baba22b4d9f3a0bfbc6616afe758d1a5_668_50_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bac6838076986bbf79d099a77fb64bc1_311_604_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bc6fc1243f4f8497e54806deddbcbcb6_742_1744_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bcff8c2dbad73faa2cbe16c884e04888_112_754_1394.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bd48ad746f4783a4ddf0cede22556b3a_19_4_140.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bdba9888cb464045d576cd7b602b6677_331_1508_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bdbaa4deac06c9692d3f81fa4b9f8784_260_1762_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bdbd6be56dc60ffec824287be91b16bc_246_0_1267.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bdf8463743423ee0a302941aba5d4f71_496_1552_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\be12b28a3f48cf4bcad356b34f99b1fc_94_0_601.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\be2b0ca14bf4742d5c4e504ac42ad839_747_1756_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\be53e37821f20c7fb929b2846364aa1e_299_230_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\be71ea278cc775cc735a7b6a58b4ee4b_170_248_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\be92aa92c30c6c92b96fe03789e07ea9_408_166_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bef6e725e640587c66f4835cd8d894e4_340_2012_1629.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bf032fdcaaeb95b542754d12da170d49_337_4_1472.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bf61a7064533b7b5e3101bddb6c5356c_67_392_682.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bf748f74ed2c8c994ae48aadc20438ad_368_566_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bf795c414587b9608cb6093bbf384d44_81_1886_1380.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bf9597c4ae464e86191dec69b5e4c666_66_1968_502.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\bffd5845bc45b9f24869edd47fa48947_243_1922_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c0283dd507d2f01c2afcdb23be7584cd_181_4_358.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c10edb4dedef3f211b51c8a47d2b5006_467_0_1259.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c14f04fb451b8fc58944c2975430329f_99_4_1182.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c19d91d5fa8031c8f3e3f98f4f88439d_299_0_1418.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c1a49bf5f73840ab688b893f7a70c41a_366_1988_1319.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c2039379a08b5ee47e05f28d0881c5d5_257_4_626.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c25e597aa5e495072bc98c8558fd0931_80_1470_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c29e20f875db893cb45f1d1ba93d8e89_45_318_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c2a3fd0ae8aaffb96eb4d51b3368f525_239_1566_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c2b48a426ba5850d0e4a2cbf158be492_348_0_513.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c36962280fabbcedf57ac97601ce59ef_340_762_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c36efe0d73dd6cc29ab0674193e86710_687_4_8.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c37ac29f60c0bdde13bc54d79dc6b284_262_672_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c3c68a4d05a478ceee947bf5fde6a3c9_67_1112_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c3cc6f17f530714a1458bebb0ce261f8_252_4_566.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c4035313bd1ef5a26d6a3db9dbc324c2_687_610_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c45deb490e4c2fbc72d1b64368251a11_93_4_316.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c4b01f139aaa3ee2d4a11d66cdc80f48_284_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c4b4634ad539c5d0efbb2a5c8981f257_628_4_412.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c4cdf7436a13f2ac8961a59cad066c31_465_316_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c5229aac93a75aa84458e5dea5d215ef_224_1964_1840.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c58f08cf51b8e91f900f39a5fbfba042_322_396_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c65b94be8d263f2d7ddc81600a2daa2d_118_48_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c65e8403eac4821e237e99cc8086e934_229_0_1932.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c6657f4a7d3c7c9c034ed42769da56dd_661_4_212.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c6930131d24c91f7ce2ed3d03462a41c_310_0_1618.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c6b51201ea42415abb0ae0a258063aa4_426_174_846.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c6cc9b1015421d3ec2f05efe559f8132_273_0_1312.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c74ac5284ec81445eee6a5dee4425164_606_180_388.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c7bd3aa4dae8cf6329681c8af4d88291_680_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c88f92f4de93b121e8981a786e0b328a_52_4_1706.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c90f0fb8affbe96541f41566e3b938a7_454_0_105.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c96711a879557d28ea4d6f00c23de5e3_439_1332_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c9979aa036c3022f696683234695cb2c_240_1968_1508.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c9bde7189c5129cf08387d5b85137805_729_960_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c9c8c423ecf4bdbad51f519f6fb31d0a_514_4_358.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\c9dbcf0195657023f884cf385419abce_327_72_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ca4024248ed20099d1d32c2d790e5039_480_874_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ca4bc26d9686b925cd2937ee2ff4f39e_467_10_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\caafd0bb1ca3fdd7c82c7916a0994a1b_326_436_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cac483f688b8bc45082fff347cfd2388_455_1912_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cae9c5fc96d309665200df671a1934e7_402_460_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cb07b3d90f0ae80a9fbd5343d88603d4_138_948_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cb12b36d60211ff4619e9b0374d66d99_540_1766_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cb1e0f5cce93c2184584122717bb2a9c_267_1058_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cb554f951b4252b7508ce7847f363f25_319_1904_318.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cbb8d8912d500bf318810da79fcdb96b_369_0_1552.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cbbeefe8e1d333ef36a265b2825a587a_415_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cbd84be4d35086997feeb1bd8432bb04_553_364_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cc723dddb1b9b2d8e0d598a49e1f95f6_215_1872_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cc772593eaa729c838d12cbe85e8ea18_592_424_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cc7dc7a72ee13e8de4359ee4361f1a40_708_4_612.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cd116dd26aaa3ab37c8278006a5a5978_583_456_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cd25ef1ca55a828d7400a9a5f1fabf2a_416_4_766.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cd3aa9de1d91b717d28e671e22dab84a_686_4_616.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cd4c839c0e9b5e74325a7752a410f944_554_0_1365.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cdec1bfd54db5cda1a5584b76bafe9ad_120_0_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cdf1e9ffecda296ead011923143f4e31_422_1332_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ce02da5e5518977e271fc79bd93fb394_413_742_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ce056990d700f7b6698483536f47be8a_535_4_1574.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ce27cc9201f2456fd641401563d65eb2_637_108_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ce65e508205f91f4048117b982bedb1e_498_498_1458.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ce88b9c03c24068140f56cd5d6b2b818_261_856_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cf26f391c8945ca6082190b9a03c837a_111_4_650.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cf677388f28ce755459962b89e9e859d_642_960_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\cfbcdff8e535013c965d6e49a02ba92d_2_1880_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d0254ff6a6d0ecbc575f45f4f0109421_99_1000_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d0b7c6547ca2a61a31cab94607fe0877_214_452_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d107f8ebe6e97c78903e3eb1ab3ce878_516_1478_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d118809001e9bb41c8336a444d60b74c_656_1952_909.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d123b2b522a1c2598622f3591831dc0e_128_4_90.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d13e19b7b854dc2d096ea25dc9d0e34a_494_4_200.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d155a72310d452e32cfc2693659d3669_382_918_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d175b7cb5e03a067d751865f3729bc78_765_1748_210.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d1846fc3a879e0ba6f52d904f14dca15_279_1222_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d20f8f5048b1d14910c49518e2b19a31_172_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d29e3d187754ad7640d088eeebaeac18_496_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d29e4aa84674a1372827dd96bbfe0e76_128_4_828.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d2a1a384a2e0f875d49779dcd489677a_400_1942_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d2fa67c000e10f8cdb3b2eea9b2be315_629_350_1713.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3088f5838de5c3f8e6c28441c4ea145_144_1542_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d328446140e1b2ff81076b5045cbbd90_767_4_422.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d33dcf6b70c1030550d7401dc09b8027_174_1956_923.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d369e5ea3cea61d253ca9d99f6cd5e1b_137_1010_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d36bb3e5401ab35e76405c328bf08c54_562_1034_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3718cd6399e580f53811629cc9a3da1_365_550_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d391dd1f7dcda130b9774f35cec45a48_725_4_1070.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3ae5aa102adfaca01562266e900944d_112_680_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3d3304ea1c9f5acd6314d7eeaa220a7_489_4_1004.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3dd7241d26295969a0d4eaf95750636_415_4_482.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3e878a58112539125ad063085981fd7_2_1356_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3eb97012c81988acfd4047dcf8fbcca_660_868_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d3f1d201980d45c08fad6a7efc5ef177_650_1096_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d4131d3e3dc4030381fbccae45c9452c_287_1952_483.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d452a24e88d8aed21fbbc670aa477341_666_1660_1018.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d4788e572d05865477595f5d8b3ca0f3_429_1288_1218.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d4ac121a00bd1aa4e28a49723409cadc_192_380_149.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d4cb158092ccd71901d90d6d8d09fcd2_271_1964_1284.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d4e50ae72f2b23ef0aaa7ed9e5b0fb51_352_836_622.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d4f79de159174eab31c64055919a1d15_446_4_1346.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d53595b71eed2dbb38364daf8f0ea0b1_547_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d5567a5ad1cc569fea4460980d2b091d_68_334_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d56785ab071ce67e3ed043a35fd39369_627_246_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d56f146aadde116ebbf28599e5bc4668_546_4_338.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d5e2b5b0e016b93c4e144ed1f371b570_236_0_1336.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d6308d7c3cfd329a2d496c2a3bc94a64_399_944_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d65307f04b1689fad032497e5915bddf_714_0_704.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d676f6543564be0c2ae15f8509f6213a_294_332_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d6ad226ad708e3a0d268794d56235319_284_2000_293.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d743fd9d52a6acc1b7a07caade8a426c_501_4_534.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d74b585f5828b0f7e185b2272b97853c_135_4_1304.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d754dab34aa860e74868273883fcdb6d_294_0_1771.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d771c047315ca419365aa463a838a9f2_648_0_1116.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d7ab13831de336021101ed2fc8f4ebc9_495_566_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d83e5446f6099bc140274a91afcf5118_64_1756_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d84f95598e7b9a10e8461ba2f207fdbe_364_606_1265.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d8acf1ba48e02f5a15017cd0cc83dbb8_322_400_306.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d8b175c21a2e9182b617b1d748d1e404_364_892_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d8d999ad9df583680c7058ebb6418101_313_4_1006.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d8fff9d9e6d234af5c02d63e00948503_498_340_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d9285dc5edc84305842856fd09ea6140_660_4_780.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d96e6ecf7aceb095767748760e97a449_497_850_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d9919a6eaefef9914328ff815b7a81cf_133_398_1850.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d997f1bd21cb6d99ca44e33ca1fc0337_291_1492_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\d9e117fb2fac4bfc1e4e4ab1e6f87bd1_374_1098_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\da020249dd16a7809830f129aeb4b2ac_339_732_1963.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\da0a93960bf3b5504030a6c3739c537b_252_4_478.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\da1a00b6ad57e2a5b2532175146cb7a9_7_1418_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\da1d36bb107588210be4da2363cebfff_614_110_236.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\da2d50914a902c1e4ce3a34edd50bfdf_129_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\da403018241a10b40db81b431140d6bc_482_4_900.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\da8c934b76b878f66cc804ea67bdb618_302_704_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\daaa096bf371538a3f95c477989f018e_13_258_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dacdf4307f5f6b5f1019e8f687921ec4_144_1702_1319.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\db32dfe4490b647e21a43b1e008254d7_130_92_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\db52eb06728d01dfab88943e77dc2506_94_508_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\db6004e888a865e49de6d88929a06bb6_194_4_918.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dbae229f5f8af1392503bc5f75d665f7_454_1586_1408.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dc5838b5f9db7090e82d721a9bc5a281_84_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dc6bb7696e9c0869640bf94f8b1678d3_346_1768_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dc8a99c836ce5a9292e572dab8f0683e_417_370_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dccd5ef9b73ea6f8ad0987b7d3680449_18_598_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dce9397e2b5953f1ff36ddee0b454521_281_2002_1602.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dcfedbda08546cef91f844db6f2537b5_371_0_281.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dd8e4e86ee7cd28b735c5f2d881fe2b1_319_1680_1836.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ddb54ff08773823013115587decba390_66_4_1188.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dddb8305127055cd18b940d7f84000a6_719_1320_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\de832d0025f72e8f7f762917bb1dc2e8_322_1952_349.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\de8c1c710be53b804de8c0726d993d05_204_1886_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dea4a6c1367f2bb5506573ee515ee73d_255_1964_316.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\decc9ee357e6b77a11556c73842cadc1_321_1292_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dee4170c6e639551b23f0aaeeb26dbeb_222_4_474.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\defd79865f0019b0ffd2a348386faae6_267_1934_1538.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\df201b0c20756a7bc08dd5f532e503bb_449_150_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\df7165c9f684c0d6556ba8d67a83e68d_109_1186_1580.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\df8a3fd6e6a523326310f619a65c431f_604_306_258.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dfb6c5e5bbcd10cde6e9982307f6a5c6_54_2000_1552.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dff860b01799f27c895985a99dd4eee1_172_0_1803.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\dffebb995831c1794562d621f1973996_23_0_1281.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e02d5a3a060a7be7f24e082a9bc5709e_92_1960_1846.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e08c2054b581300879e4ab3e61161db5_624_602_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e0fc0360c2bfddb1db6daf9889dafa46_558_0_1532.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e1269583d01054e001f9b4dfe1144509_614_692_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e13ef9a858770e6b165064022cd1e2d7_250_1984_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e1a69018c2d732a1482a2ceb3c29d04b_5_4_27.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e1ea447a44ebcb15570688c63d1844cc_706_1952_1641.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e2543cf9a380b958fff32d211423049c_54_4_1201.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e2a8f12701a709b6607274209726b571_82_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e2ae76e11f02a67de608c4484c4b1c42_14_4_20.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e2f3e568686d011412661c51ca30d363_443_430_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e31312f7a2a67870f12ea07b99e0281f_595_4_1110.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e4a46a48c4bc5edc99f6311d40a7cb50_76_848_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e4cdf5720ba0dddd555f944de4344adf_310_1964_182.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e4e09c48adf019573d358b9f3231ae60_506_1960_907.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e503c231ef01d984f6eac6f9fd0ecd82_742_1506_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e52b2162e50dcb0559659eff9b9563c7_488_0_1836.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e535c565e36432b2fb9d89d2b5f2ac54_212_932_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e5ca61fdab00e50e6860659685f356eb_721_4_268.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e654a8784402b7c0041532e2c8088028_705_4_478.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e65f1882241a5718f7d6d15649be7c6f_389_0_1620.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e67a2151c6f83f5ffe3f8c42563900c9_563_4_952.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e69b8edf495607ed4a0560e37b367d94_529_1418_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e6ce7a0ec0dc6b66916916847045865b_435_1618_974.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e72bd259b3dd8841698aafb352b51c79_109_1324_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e74dc8f5ba4032aa139f872b56876e9a_262_692_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e7c92ec4b965433880b17a7b95d3e0a2_766_1612_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e8380c1920564386432d78d4d055a815_229_4_620.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e87bcf24366ac1ee8a2e9591e2789c71_54_2004_762.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e89b536b4dd8326434c4095cbce33e61_367_4_1482.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e8b44a8c52a5366f49e959a8291b7c81_687_4_1246.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e8c53ae649cb3fc314d1b1d50c356a7e_480_1096_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e921ee90eb5ea80b37e0afd051e42272_728_4_340.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e9a3e206e6a2380a096db8e95609c972_70_1856_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e9e16c853243c389b5e55852bb3871fb_193_1298_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\e9ff5e3ee2f0d3221530db05463c1223_317_1970_198.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ea03dfe9ea1d87fbe3b9f9ce39c866be_193_4_1244.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ea06f01cc0173cdcca38e2945b9d33da_626_1622_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ea74b0a9f6e0286d9e4974db8fd3b9e9_772_0_1746.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eaaf41166126881484aea062c6769b8b_206_1704_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eabd29a7177b61a578f8496c44ba4bf7_246_436_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eafb7476f5692529f69b7bdafc4b300a_441_1084_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eb1f66d4fa27f5ea4ded5cdd81c4f780_497_1178_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ebea1f5924b640316dfa59ea8b3218d8_529_4_578.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ebef69b8a2f1033229275361d7713b06_348_986_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ec4ba54bf23ac158ce67ad51dc382aab_490_1102_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ec73dddcbbcdfdbb18a367471e1b16d4_197_404_3.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ec9e3c2f3a3c572476a870bf9b2b96a9_346_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ece38186307369a8954d42d63a0be9b6_171_34_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ed4c89b083b0b9dfbc2e8d5479501ac6_580_1470_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\edb5b6820cc538f688b897acb6272c91_277_2004_1387.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\edd7909ff34b206c5405adcf38b87f76_339_4_960.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ee48b47e545070ac173da548ad409ac8_639_0_1906.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ee637bc364335d4d4a26a40411a3dff8_297_4_262.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ee6fc1dbb0eddf935617be962d0fa476_214_4_1910.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ee8141db59e653d7148e2caf9eb6244e_674_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eead619ac115ef2a57709f20d6675bc5_73_1972_268.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eed3b0ed8bf903bb4db409128b54a8a9_463_504_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eedbf4328ff5c1aa79d5d7e0e2ad9575_752_178_1963.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ef1781a182fe3fffb7a4ab02352c51be_587_1408_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ef5b785836406a495a07513474758388_417_132_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ef96b90a71ab053573cdaf74103f4efc_727_1994_926.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\efa1a387969feaff610875282ee7bce5_56_4_740.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\efa59c3cb1be7fd7583aa6eee630a64b_392_4_1164.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\efe753d6c7940f51017d612bdafd7619_255_332_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\eff9c70f93f52a7ea0f6d678f71bab6a_258_4_1566.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f000f3651d4d33a5b6dceffea483f971_170_1968_1490.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f02c02571be1761396456a22dd674fbe_457_1996_604.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f09a089185666825f29cfd201fef8941_527_0_694.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f0cb497e8cdba5bc8b491be67809c046_655_1616_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f0dcd0020b12ead6500dc6b67a65360c_166_4_1216.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f0e1e34ebea7271c4967bf4f3a7b7690_729_0_972.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f1b8a066e143a8cb930cbd84daf15369_365_4_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f2409d444fcb94b55aeb70b3f7334cbc_544_1006_1596.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f27a454ef10dfd4c8aee166297008c66_274_1996_1260.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f27a88f12cfd5d36ef76662ab30079a3_299_1970_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f28507eb40a49a85b6bdc8e628150632_745_376_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f29076b059f8526d8d94765945c2f245_280_0_1256.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f292249f5db41add94f6de445e59e19d_471_1998_12.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f33ff40d7902939790eff1dc327dda94_222_136_1252.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f346b22d6a7dde59b4f03416c95da21d_68_1698_1202.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f382bcb56a1f7eaa023298f1e1daf4ed_292_944_390.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f38532460c5fe3f38f2ca6672f4a434f_219_1422_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f45117da706789f01c4e059820d8b182_675_176_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f45e7bb872949698e119b0b00328c929_296_1488_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f4d323795bc44d18ec879fe8dce4d8ae_631_0_1980.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f51c6369cfd180db5298f4e198254274_169_1908_998.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f57215ad197f52bce5b00005c82d3a21_344_4_1512.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f5748836c2e1dd8bf4b15e47e7ba43cb_554_1162_28.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f5929d4468dc8d00103e21e589fabcc3_704_898_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f5e1d7ca64e337ec4028d7eba02847d7_340_514_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f61942c934f192fad439c66c7b819aa7_416_4_254.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f6202d3793a183f0ffe97baac95671d1_352_1072_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f628acaa0af444ec3dc50060dc98b04d_684_204_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f64ee452af4acba1ab1dfa4b15790ff3_382_10_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f6885595e4e1508c485048f78bb98771_97_1940_38.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f688ff8ea34727963da92cbbeb908365_262_412_1118.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f6d2360279ed5820c50182fdab171a8d_197_4_678.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f6d7603dd8d30b267d8bb35942cd0eed_310_2000_239.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f7119df199574af1fc7cce643e082d9f_322_434_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f7b865372836ea0a8ff8a1d75a1abab0_429_1308_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f83e31c0933fb11fde07a34ce3569408_54_292_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f8617a6a5c4bc06671a2e2db9b87645c_747_4_1204.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f86548da2ca50ac712f399f2994640c4_704_1868_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f8ca4b516d68c135299c2624fe254a31_202_1990_286.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f9347005977afe4cc60a17f34b6895f0_530_1320_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f993610ac06751223d6e3cb31c96cff5_207_0_869.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f9ded7f8c1168a88aaba719ee66f9057_416_554_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\f9dfa1182e52567f4652b0438fa16330_541_0_1370.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fa44d57dcdb881aa016b11d5f621471b_227_0_719.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\faa3b8e73e73706962c3ea0557bc310f_736_498_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fae8ca43bb1edcb741c5325a0cd82617_665_810_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fb35b9d82a8e6f3d7288d9ebc58c104f_122_1124_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fb51e043802eddec19b9612192a4cba8_511_0_1851.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fb5494ddae5851e9bf904f3f4945d8b8_156_0_1269.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fb6508768b7e284759ca38223960e53a_228_4_1122.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fb897637085de1fcd3ef86bd14d6d8a8_408_1858_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fc52ab480369863005e1d6242c80cb49_649_4_1858.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fc6ec54e00ba2b7d99841965ec1a2dfe_667_0_115.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fd318c868782f42e0a95f75bd17d5bb5_94_4_1415.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fd554991442c7ee497ac078d439b45ab_616_1998_1418.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fd9a4a7af2cf1cc25f3afdf5bf8d53dd_613_4_1418.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fe3f23286969784d49dd60bfefa2c5ec_134_1664_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fe488d7f1aac1b0283dddff8e25f6e36_321_1424_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fe61090f51a61daf113fc37ed3c3a77f_46_1848_1.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fe8d4918da36ebd3c5f6107dbf0bff49_396_2_2.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fecdfd0025909f54c59b12dd3ecb6217_20_1990_1674.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fef5f395e3c844d73da30fffe36f589a_692_1732_0.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\fef6662a05c9d7441a837320d7d8212f_656_4_394.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ff3ec77f0a8df3f65cbcfa7bb744451a_422_466_1967.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ff47b2d609b65f5b4425fb8966f243db_443_4_1614.bmp",
"C:\\Unpack\\qc_train\\cutted$01_NeedTrain\\ffd631d6d8911cd0a60f108cad44491d_62_4_2.bmp",
};


		// 测试

		ncnn::Net mobilev3;
		mobilev3.opt.use_vulkan_compute = true;

		mobilev3.load_param("mobilenetv3-large-1cd25616.pth.sim.param");
		mobilev3.load_model("mobilenetv3-large-1cd25616.pth.sim.bin");


		int w = m.w;
		int h = m.h;
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(m, ncnn::Mat::PIXEL_BGR2RGB, w, h, 224, 224);

		// transforms.ToTensor(),
		// transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		// R' = (R / 255 - 0.485) / 0.229 = (R - 0.485 * 255) / 0.229 / 255
		// G' = (G / 255 - 0.456) / 0.224 = (G - 0.456 * 255) / 0.224 / 255
		// B' = (B / 255 - 0.406) / 0.225 = (B - 0.406 * 255) / 0.225 / 255
		const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
		const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
		in.substract_mean_normalize(mean_vals, norm_vals);


#if NCNN_VULKAN
		{
			cout << "ncnn warm up" << std::endl;
			// warm up
			ncnn::Extractor ex = mobilev3.create_extractor();
			ex.input("input", in);
			ncnn::Mat out;
			ex.extract("output", out);
			std::vector<float> cls_scores;
			cls_scores.resize(out.w);
			for (int j = 0; j < out.w; j++)
			{
				cls_scores[j] = out[j];
			}
			auto p = soft_max(cls_scores);
			print_topk(p, 5);
		}
#endif



		ncnn::VkCompute cmd(mobilev3.vulkan_device());
		mobilev3.opt.staging_vkallocator = mobilev3.vulkan_device()->acquire_staging_allocator();
		mobilev3.opt.blob_vkallocator = mobilev3.vulkan_device()->acquire_blob_allocator();
		mobilev3.opt.workspace_vkallocator = mobilev3.opt.blob_vkallocator;
		mobilev3.opt.blob_allocator = new ncnn::PoolAllocator;
		mobilev3.opt.workspace_allocator = new ncnn::PoolAllocator;
		mobilev3.opt.lightmode = true;

		//{
		//	ncnn::VkMat vkIn;
		//	ncnn::VkMat vkIn2;
		//	cmd.record_upload(in, vkIn, mobilev3.opt);
		//	cmd.record_upload(in, vkIn2, mobilev3.opt);
		//	cmd.submit_and_wait();
		//	{
		//		cout << "VK warm up" << std::endl;
		//		ncnn::Extractor ex = mobilev3.create_extractor();
		//		ex.input("input", vkIn);
		//		ncnn::Mat out;
		//		ex.extract("output", out);
		//		std::vector<float> cls_scores;
		//		cls_scores.resize(out.w);
		//		for (int j = 0; j < out.w; j++)
		//		{
		//			cls_scores[j] = out[j];
		//		}
		//		auto p = soft_max(cls_scores);
		//		print_topk(p, 5);
		//		//ncnn::VkMat out;
		//		//ex.extract("output", out, cmd);
		//	}
		//}

		std::vector<ncnn::Mat> cachesMat;
		for (auto file : files)
		{
			auto path = string2wstring(file);
			auto img = load(path);
			ncnn::Mat resize = ncnn::Mat::from_pixels_resize(img, ncnn::Mat::PIXEL_BGR2RGB, img.w, img.h, 224, 224);
			resize.substract_mean_normalize(mean_vals, norm_vals);
			cachesMat.push_back(resize);
			//if(cachesMat.size() > 5)
			//	break;
		}

		//auto devkit = ncnn::get_gpu_device(0);
		//int max_batch_size = devkit->info.compute_queue_count();
		//#pragma omp parallel for num_threads(max_batch_size)

		clock_t start = clock();
		clock_t end = clock();
		double time = 0;




		start = clock();
		// 批量把图片塞入 GPU 
		std::vector<ncnn::VkMat> cachesVkMat(cachesMat.size());
		for (int i = 0; i < cachesMat.size(); ++i)
		{
			cmd.record_upload(cachesMat[i], cachesVkMat[i], mobilev3.opt);
		}
		cmd.submit_and_wait();
		// 再推理
		for (int i = 0; i < cachesVkMat.size(); ++i)
		{
			ncnn::Extractor ex = mobilev3.create_extractor();
			ex.input("input", cachesVkMat[i]);
			ncnn::Mat out;
			ex.extract("output", out);
			//if (i < 5)
			//{
			//	std::vector<float> cls_scores;
			//	cls_scores.resize(out.w);
			//	for (int j = 0; j < out.w; j++)
			//	{
			//		cls_scores[j] = out[j];
			//	}
			//	auto p = soft_max(cls_scores);
			//	cout << i << "out " << std::endl;
			//	print_topk(p, 5);
			//}
		}
		end = clock();
		time = (double)(end - start) / CLOCKS_PER_SEC;
		cout << "ncnn [224,224] X " << cachesVkMat.size() << " GPU mat time << " << time << "s, avg-cost " << time / cachesMat.size() << " s, " << cachesMat.size() / time << " fps" << endl;//输出运行时间








		start = clock();
		// 直接用 CPU Mat 输入推理
		for (int i = 0; i < cachesMat.size(); ++i)
		{
			ncnn::Extractor ex = mobilev3.create_extractor();
			ex.input("input", cachesMat[i]);
			ncnn::Mat out;
			ex.extract("output", out);
		}
		end = clock();
		time = (double)(end - start) / CLOCKS_PER_SEC;
		cout << "ncnn [224,224] X " << cachesMat.size() << " CPU mat time << " << time << "s, avg-cost " << time / cachesMat.size() << " s, " << cachesMat.size() / time << " fps" << endl;//输出运行时间



		//const int loop_count = 1000;
		//start = clock();
		//for (int i = 0; i < 1000; ++i)
		//{
		//    ncnn::Extractor ex = mobilev3.create_extractor();
		//    ex.input("input", in);
		//    ncnn::Mat out;
		//    ncnn::VkMat vkOut;
		//    ex.extract("output", out);
		//	if(i == 1)
		//	{
		//        std::vector<float> cls_scores;
		//        cls_scores.resize(out.w);
		//        for (int j = 0; j < out.w; j++)
		//        {
		//            cls_scores[j] = out[j];
		//        }
		//        auto p = soft_max(cls_scores);
		//        print_topk(p, 5);
		//	}
		//}
		//end = clock();
		//time = (double)(end - start) / CLOCKS_PER_SEC;
		//cout << "ncnn [224,224] X " << loop_count << " time << " << time << "s, avg-cost " << time / loop_count << " s, " << loop_count / time << " fps" << endl;//输出运行时间
	}
	return 0;
}