#include "loghelper.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <mutex>

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#define _WIN_
#endif

#ifndef _WIN_
static int _vscprintf(const char* format, va_list pargs) {
	int retval;
	va_list argcopy;
	va_copy(argcopy, pargs);
	retval = vsnprintf(NULL, 0, format, argcopy);
	va_end(argcopy);
	return retval;
}
#endif

// 获得当前时间，以"[20190101 00:00:00] \0"格式存入buff，buff长度为1+8+1+8+1+1+1=21
static void getTime(char* buff)
{
	time_t tt;
	time(&tt);
	tt = tt + 8 * 3600;  // transform the time zone
	tm* t = gmtime(&tt);
	t->tm_year += 1900;
	t->tm_mon += 1;
	sprintf(buff, "[%d%02d%02d %02d:%02d:%02d] \0",
		t->tm_year,
		t->tm_mon,
		t->tm_mday,
		t->tm_hour,
		t->tm_min,
		t->tm_sec);
}

#ifdef _WIN_
#include "windows.h"
// see https://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
// see https://learn.microsoft.com/en-us/windows/console/console-screen-buffers#span-idwin32characterattributesspanspan-idwin32characterattributesspancharacter-attributes
#define RED     12      /* Red */
#define GREEN   10      /* Green */
#define BLUE    9      /* Blue */
#define YELLOW  14      /* Yellow */
#define MAGENTA 13      /* Magenta */
#define CYAN    11      /* Cyan */
#define RED_BG    RED | GREEN |BLUE | (RED << 4)
#else
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define RED_BG  "\033[1m\033[7m\033[31m"      /* Bold Red 反向 */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#endif

log_helper log_helper::log;


static void outputToConsole(const log_helper::enum_level level, const std::string time, const std::string level_str, const std::string log, const std::string prefix)
{
	// colorful
#ifdef _WIN_
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	int color = 0;
	switch (level) {
	case log_helper::enum_level::info: color = CYAN; break;
	case log_helper::enum_level::warning: color = YELLOW; break;
	case log_helper::enum_level::error: color = RED; break;
	case log_helper::enum_level::fatal: color = RED_BG; break;
	case log_helper::enum_level::debug: color = GREEN; break;
	}
	SetConsoleTextAttribute(hConsole, color);
	std::cout << time << "\t" << level_str << "\t\t" << prefix << log << std::endl;
	SetConsoleTextAttribute(hConsole, RED | GREEN | BLUE);
#else
	std::string color = "";
	switch (level) {
	case log_helper::enum_level::info: color = CYAN; break;
	case log_helper::enum_level::warning: color = YELLOW; break;
	case log_helper::enum_level::error: color = RED; break;
	case log_helper::enum_level::fatal: color = RED_BG; break;
	case log_helper::enum_level::debug: color = GREEN; break;
	}
	std::cout << RESET << color << time << "\t" << level_str << "\t\t" << prefix << log << RESET << std::endl;
#endif
}

static void outputToFile(const std::string file_path, const std::string time, const std::string level, const std::string log, const std::string perfix)
{
	//CleanupTxt(file_path);
	std::ofstream in;
	in.open(file_path, std::ios::app);
	in << time << "\t" << level << "\t\t" << perfix << log << std::endl;
	in.close();
}


static int CountTxtLines(const std::string file_path)
{
	std::ifstream read_file;
	int n = 0;
	std::string tmp;
	read_file.open(file_path, std::ios::in);//ios::in 表示以只读的方式读取文件
	if (read_file.fail())//文件打开失败:返回0
	{
		return 0;
	}

	//文件存在
	while (std::getline(read_file, tmp))
	{
		n++;
	}
	read_file.close();
	return n;
}

static void CleanupTxt(const std::string file_path)
{
	auto l = CountTxtLines(file_path);
	if (l < 10000)
	{
		return;
	}

	int index = l - 800;

	std::ifstream read_file;
	int n = 0;
	read_file.open(file_path, std::ios::in);//ios::in 表示以只读的方式读取文件
	if (read_file.fail())//文件打开失败:返回0
	{
		return;
	}


	std::string tmp;
	std::vector<std::string> vecContent;
	while (read_file)
	{
		std::getline(read_file, tmp);
		n++;
		if (n > index)
			vecContent.emplace_back(tmp);
	}
	read_file.close();


	std::ofstream write_file(file_path, std::ios::out);
	if (write_file.fail())
		return;

	auto iter = vecContent.begin();
	for (; vecContent.end() != iter; ++iter)
	{
		write_file.write((*iter).c_str(), (*iter).size());
		write_file << '\n';
	}
	write_file.close();
}

void log_helper::set(log_helper::enum_level print_level, log_helper::enum_level write_level)
{
	print_level_ = print_level;
	write_level_ = write_level;
}

void log_helper::init(std::string log_file_path, log_helper::enum_level print_level, log_helper::enum_level write_level)
{
	log_file_path_ = log_file_path.c_str();
	set(print_level, write_level);
}

void log_helper::clear() const
{
	FILE* fp_write = nullptr;
	fp_write = fopen(this->log_file_path_.c_str(), "w+");
	char time[21] = { 0 };
	getTime(time);
	fputs(time, fp_write);
	fputs("\r\n", fp_write);
	fclose(fp_write);
}


void log_helper::output(const log_helper::enum_level level, const char* format, ...) const
{
	static std::mutex mtx; // 保护counter
	if (this->print_level_ > level && this->write_level_ > level)
	{
		return;
	}

	std::string var_str;

	va_list	ap;
	va_start(ap, format);
	const int len = _vscprintf(format, ap);
	if (len > 0)
	{
		std::vector<char> buf(len + 1);
		vsprintf(&buf.front(), format, ap);
		var_str.assign(buf.begin(), buf.end() - 1);
	}
	va_end(ap);

	char time[21] = { 0 };
	getTime(time);
	std::string str_level = "DEBUG";
	switch (level) {
	case enum_level::info: str_level = "INFO"; break;
	case enum_level::warning: str_level = "WARNING"; break;
	case enum_level::error: str_level = "ERROR"; break;
	case enum_level::fatal: str_level = "FATAL"; break;
	case enum_level::debug: break;
	}

	if (this->print_level_ <= level)
	{
		mtx.lock();
		outputToConsole(level, time, str_level, var_str, this->prefix_);
		mtx.unlock();
	}
	if (this->write_level_ <= level)
	{
		mtx.lock();
		outputToFile(this->log_file_path_, time, str_level, var_str, this->prefix_);
		mtx.unlock();
	}
}