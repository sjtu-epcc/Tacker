#include "Recorder.h"
#include "TackerConfig.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

bool cmp(const std::pair<int, std::vector<float>>& a, const std::pair<int, std::vector<float>>& b) {
    return a.first < b.first;
}

void Recorder::text() {
    std::string filename = "recorder_output.txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "cannot open file: " << filename << std::endl;
        return;
    }

    for (auto& task_pair : task_time_map) {
        int taskId = task_pair.first;
        auto& times = task_pair.second;
        auto& kernelIds = task_kernel_map[taskId];
        auto& taskName = task_name_map[taskId];

        std::cout << "taskID: " << taskId << ", taskName: " << taskName << "\n";
        file << "taskID: " << taskId << ", taskName: " << taskName << "\n";

        float totalTaskTime = 0.0;
        for (size_t i = 0; i < times.size(); ++i) {
            int kernelId = kernelIds[i];
            float time = times[i];
            totalTaskTime += time;
            auto& kernelName = task_kernel_name_vec[taskId][i];

            file << "    kernelID: " << kernelId << ", kernelName: " << kernelName
                 << ", execTime: " << time << "ms\n";
        }

        file << "    task running time: " << totalTaskTime << "ms\n\n";

        // std::cout << "Task: " << taskName << ", taskID: " << taskId << ", task running time: " << totalTaskTime << "ms\n";
    }
}