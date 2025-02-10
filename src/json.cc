#include "json.h"
#include "Logger.h"
boost::property_tree::ptree ptr;

boost::property_tree::ptree common_ptr;

extern Logger logger;

int get_kernel_info(const std::string &kernel_name, const std::string &key){
    auto val = ptr.get_child_optional(kernel_name);
    if (!val){
        return JSON_NOT_FOUND;
    }
    auto val2 = val.get().get_child_optional(key);
    if(val2){
        return val2.get().get_value<int>();
    } else {
        return JSON_NOT_FOUND;
    }
}

float fget_kernel_info(const std::string &kernel_name, const std::string &key){
    auto val = ptr.get_child_optional(kernel_name);
    if (!val){
        return JSON_NOT_FOUND;
    }
    auto val2 = val.get().get_child_optional(key);
    if(val2){
        return val2.get().get_value<float>();
    } else {
        return JSON_NOT_FOUND;
    }
}

std::string sget_kernel_info(const std::string &kernel_name, const std::string &key){
    auto val = ptr.get_child(kernel_name).get_child_optional(key);
    if(val){
        return val.get().get_value<std::string>();
    } else {
        return "";
    }
}

void read_json(const std::string &filename)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Error: file not found" << std::endl;
    exit(1);
  }
  std::string str;
  std::string line;
  while (std::getline(file, line))
  {
    str += line;
  }
  file.close();
  ptr = boost::property_tree::ptree();
  std::stringstream ss;
  ss << str;
  boost::property_tree::read_json(ss, ptr);
  logger.INFO("Read json file: " + filename);
}

void add_kernel_info(const std::string &kernel_name, const std::string &key, int value){
    ptr.put(kernel_name + "." + key, value);
}

void build_json(const std::string &filename)
{
  std::ofstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Error: file not found" << std::endl;
    exit(1);
  }
  boost::property_tree::write_json(file, ptr);
  file.close();
}

void read_common_json(const std::string &filename)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Error: file not found" << std::endl;
    exit(1);
  }
  std::string str;
  std::string line;
  while (std::getline(file, line))
  {
    str += line;
  }
  file.close();
  common_ptr = boost::property_tree::ptree();
  std::stringstream ss;
  ss << str;
  boost::property_tree::read_json(ss, common_ptr);
  logger.INFO("Read json file: " + filename);
}

long long geti(int count, const char* keys...)
{
    va_list args;
    va_start(args, keys);
    
    boost::property_tree::ptree tmp = common_ptr;
    long long result = 0;
    std::vector<const char*> key_list;

    for (int i = 0; i < count; i++)
    {
        const char* key = (!i) ? keys : va_arg(args, const char*);
        key_list.push_back(key);

        if (tmp.get_child_optional(key))
        {
            tmp = tmp.get_child(key);
            if (i == count - 1)
            {
                result = tmp.get_value<long long>();
                va_end(args);
                return result;
            }
        }
        else {
            // Handle the case where the key is not found
            printf("key chain: ");
            for (auto key : key_list)
                printf("%s -> ", key);
            printf("END, not found!\n");
            exit(1);
        }
    }
}