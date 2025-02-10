#include "Logger.h"

Logger::Logger(const std::string& filePath, bool log2Console, bool log2File)
    : log2Console(log2Console), log2File(log2File) {
    if (log2File) {
        // 递归判断或者创建文件夹
        std::string::size_type pos = filePath.find_last_of('/');
        std::string dir = filePath.substr(0, pos);
        if (access(dir.c_str(), 0) == -1) {
            mkdir(dir.c_str(), 0777);
        }
        logFile.open(filePath, std::ios::app);
        if (logFile.is_open()) {
            std::cout << "Logging to file: " << filePath << std::endl;
        } else {
            std::cerr << "Failed to open log file: " << filePath << std::endl;
            log2File = false;
        }
    }
}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    std::string formattedMessage = getTimeStamp() + " - ";

    switch (level) {
        case LogLevel::INFO:
            formattedMessage += "[INFO] ";
            break;
        case LogLevel::WARNING:
            formattedMessage += "[WARNING] ";
            break;
        case LogLevel::ERROR:
            formattedMessage += "[ERROR] ";
            break;
        default:
            formattedMessage += "[DEBUG] ";
            break;
    }

    if (log2Console) {
        std::cout << formattedMessage << message << std::endl;
    }

    if (log2File && logFile.is_open()) {
        logFile << formattedMessage << message << std::endl;
    }

    return ;
}

std::string Logger::getTimeStamp() {
    // Get current timestamp as a string
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}