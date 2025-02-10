// Logger.h
#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>
#include "TackerConfig.h"

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    // Constructor with log file path and log levels
    Logger(const std::string& filePath, bool log2Console = true, bool log2File = false);

    // Destructor to close the log file
    ~Logger();

    // Log messages
    void log(LogLevel level, const std::string& message);

    void INFO(const std::string& message) {
        log(LogLevel::INFO, message);
    }

    void WARNING(const std::string& message) {
        log(LogLevel::WARNING, message);
    }

    void ERROR(const std::string& message) {
        log(LogLevel::ERROR, message);
        exit(1);
    }

    void DEBUG(const std::string& message) {
        #ifdef BUILD_TYPE
        if (std::string(BUILD_TYPE) == "DEBUG") {
            log(LogLevel::DEBUG, message);
        }
        #endif
    }

private:
    std::ofstream logFile;
    bool log2Console;
    bool log2File;

    // Get current timestamp as a string
    std::string getTimeStamp();
};