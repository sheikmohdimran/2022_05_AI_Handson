! pip install transformers datasets neural_compressor -qqq

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, Trainer, default_data_collator
from datasets import load_dataset, load_metric

task_to_keys = {"sst2":("sentence",None)}
model_name = "philschmid/MiniLM-L6-H384-uncased-sst2"
task ="sst2"
padding = "max_length"
max_seq_length = 128
max_eval_samples = 200
metric_name = "eval_accuracy"
data_collator = default_data_collator
sentence1_key, sentence2_key = task_to_keys[task]

dataset = load_dataset("glue", task, split='validation')
metric = load_metric("glue",task)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def preprocess_function(examples):
  args = ((examples[sentence1_key],)if sentence2_key is None else (examples[sentence1_key],examples[sentence2_key]))
  result = tokenizer(*args,padding=padding, max_length=max_seq_length,truncation=True)
  return result

eval_dataset = dataset.map(preprocess_function, batched = True)
eval_dataset = eval_dataset.select(range(max_eval_samples))

def compute_metrics(p:EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions,tuple) else p.predictions
  preds = np.argmax(preds,axis =1)
  result = metric.compute(predictions=preds, references = p.label_ids)
  if len(result) >1:
    result['combined_score'] = np.mean(list(result.values())).item()
  return result

trainer = Trainer(
    model = model,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,
    tokenizer = tokenizer,
    data_collator = data_collator 
)
trainer.args.per_device_eval_batch_size = 4

def take_eval_steps(model,trainer,metric_name):
  trainer.model = model
  metrics = trainer.evaluate()
  return metrics.get(metric_name)

def eval_func(model):
  return take_eval_steps(model, trainer, metric_name)

# Commented out IPython magic to ensure Python compatibility.
# %%writefile quantization.yml
# model:                                               # mandatory. used to specify model specific information.
#   name: bert 
#   framework: pytorch                       # mandatory. possible values are tensorflow, mxnet, pytorch, pytorch_ipex, onnxrt_integerops and onnxrt_qlinearops.
# 
# quantization:
#   approach: post_training_dynamic_quant              # optional. default value is post_training_static_quant.                                   
# 
# tuning:
#   accuracy_criterion:
#     relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
#   exit_policy:
#     timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
#     max_trials: 1200
#   random_seed: 9527                                  # optional. random seed for deterministic tuning.

from neural_compressor.experimental import Quantization, common

quantizer = Quantization('./quantization.yml')
quantizer.model = common.Model(model)
quantizer.tokenizer = tokenizer
quantizer.calib_dataloader = common.DataLoader(eval_dataset)
quantizer.eval_func = eval_func

q_model = quantizer()

q_model.save('./saved_model')

eval_dataset = dataset.map(preprocess_function, batched = True)

def benchmark(model):
  trainer.model = model
  trainer.args.per_device_eval_batch_size = 1
  result = trainer.evaluate(eval_dataset = eval_dataset)
  throughput = result['eval_samples_per_second']
  print(f"Accuracy: {result.get(metric_name)}")
  print(f"Latency: {(1000/throughput)} ms")
  print(f"Throughput: {throughput} samples/sec")

from neural_compressor.utils.pytorch import load
model = AutoModelForSequenceClassification.from_pretrained(model_name)
quant_model = load('./saved_model',model)

benchmark(model)

benchmark(quant_model)

