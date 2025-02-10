# 遍历{model}/{model}_kernel_class内容，找出new {model}_{type}的所有type
import re
type_set = set()
num_dict = {}
model_list = ["resnet50", "vgg16", "inception3", "vgg11", "vit", "bert"]
patterns = [
    r"new {model}_(\w+)_float",
    r"new {model}_(\w+)_int32_t",
]
for model in model_list:
    model_path = f"{model}/{model}_kernel_class.h"
    with open(model_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            for pattern in patterns:
                pattern = pattern.format(model=model)
                matches = re.findall(pattern, line)
                if len(matches) == 0:
                    continue
                match = matches[0]
                type_name = match.split("_")[0]
                print(type_name)
                type_set.add(type_name)
                if model not in num_dict:
                    num_dict[model] = 1
                else:
                    num_dict[model] += 1
print("total type for all model:", type_set.__len__())
print(num_dict)

num_sum = 0
for num in num_dict.values():
    num_sum += num

print("avg kernel num:", num_sum / len(num_dict))