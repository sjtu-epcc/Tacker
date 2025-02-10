// TaskManager.h
#pragma once

#include "Task.h"
#include <vector>

class TaskManager {
public:
    TaskManager(Task* lc_task, const std::string& be_task1, const std::string& be_task2);
    void executeAllTasks(ExecutionMode mode, cudaStream_t stream);
    void execute_with_one_cd_kernel(ExecutionMode mode, cudaStream_t stream);
    ~TaskManager();

    Task* lc_task;
    std::string be_task1_name;
    std::string be_task2_name;
    int be_task_finish_num = 0;
};