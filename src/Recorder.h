// Recorder.h
#pragma once
#include <unordered_map>
#include <vector>
#include <string>

class Recorder {
public:
    void recordKernel(int taskId, int kernelId, const std::string& kernelName, 
                float executionTime) {
        task_time_map[taskId].push_back(executionTime);
        task_kernel_map[taskId].push_back(kernelId);
        task_kernel_name_vec[taskId].push_back(kernelName);
    }
    void recordTask(int taskId, const std::string& taskName) {
        task_name_map[taskId] = taskName;
    }
    void text();

private:
    std::unordered_map<int, std::vector<float> > task_time_map; // task_id -> execution_time vector
    std::unordered_map<int, std::vector<int> > task_kernel_map; // task_id -> kernel_id vector
    std::unordered_map<int, std::vector<std::string> > task_kernel_name_vec; // task_id -> kernel_name vector
    std::unordered_map<int, std::string> task_name_map; // task_id -> task_name
};
