#ifndef _LOGHELPER_
#define _LOGHELPER_
#include <complex>
#include <string>
#include <sstream>


namespace logger
{
	class log_helper
	{
	public:
		static log_helper log;
	private:
		// 禁用拷贝构造函数和赋值运算符
		log_helper(const log_helper&) = delete;
		log_helper& operator=(const log_helper&) = delete;
	public:

		enum class enum_level
		{
			debug = 0,
			info = 1,
			warning = 2,
			error = 3,
			fatal = 4
		};

		explicit log_helper(std::string log_file_path = "cpp.log", enum_level print_level = enum_level::info, enum_level write_level = enum_level::warning)
		{
			init(log_file_path, print_level, write_level);
		}

		void set(log_helper::enum_level print_level = enum_level::info, log_helper::enum_level write_level = enum_level::warning);

		void init(std::string log_file_path = "cpp.log", log_helper::enum_level print_level = enum_level::info, log_helper::enum_level write_level = enum_level::warning);


		/**
		 * \brief 清空 log 文件
		 */
		void clear() const;

		void output(const log_helper::enum_level level, const char* format, ...) const;

		std::string log_file_path_ = "cpp.log";
		std::string prefix_ = "";
		enum_level print_level_ = enum_level::info;
		enum_level write_level_ = enum_level::warning;
	private:
	};
}




using namespace logger;
#define LOG_DEBUG(fmt,...)\
logger::log_helper::log.output(logger::log_helper::enum_level::debug, fmt,##__VA_ARGS__)
#define LOG_INFO(fmt,...)\
logger::log_helper::log.output(logger::log_helper::enum_level::info, fmt,##__VA_ARGS__)
#define LOG_WARNING(fmt,...)\
logger::log_helper::log.output(logger::log_helper::enum_level::warning, fmt,##__VA_ARGS__)
#define LOG_ERROR(fmt,...)\
logger::log_helper::log.output(logger::log_helper::enum_level::error, fmt,##__VA_ARGS__)
#define LOG_FATAL(fmt,...)\
logger::log_helper::log.output(logger::log_helper::enum_level::fatal, fmt,##__VA_ARGS__)
#endif
