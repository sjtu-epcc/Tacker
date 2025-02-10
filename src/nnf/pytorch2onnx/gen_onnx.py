
# from pathlib import Path
# from transformers.convert_graph_to_onnx import convert

# Handles all the above steps for you
# convert(framework="pt", model="bert-base-cased", output=Path("./onnx/bert-base-cased-tf.onnx"), opset=11)


# way 2
from optimum.exporters.tasks import TasksManager
all_files, _ = TasksManager.get_model_files("google/vit-base-patch16-224-in21k")
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained("google/vit-base-patch16-224-in21k",from_transformers=True)