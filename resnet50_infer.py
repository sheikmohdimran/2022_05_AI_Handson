import torch
import torchvision.models as models

def inference(model, data, args):
  with torch.no_grad():
    model = torch.jit.trace(model, data)
    model = torch.jit.freeze(model)
    model(data)

    if args.num_streams > 1:
      import intel_extension_for_pytorch as ipex
      cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
      model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=args.num_streams, cpu_pool=cpu_pool)

    for i in range(100):
      model(data)

    import time
    start = time.time()
    for i in range(100):
      model(data)
    end = time.time()
    print('Inference took {:.2f} ms in average'.format((end-start)/100*1000))

def main(args):
  model = torch.load('myModel')
  model.eval()

  data = torch.rand(args.num_streams, 3, 224, 224)

  if args.intel_extension_for_pytorch or args.num_streams > 1:
    import intel_extension_for_pytorch as ipex
    model = model.to(memory_format=torch.channels_last)
    data = data.to(memory_format=torch.channels_last)
    model = ipex.optimize(model)
  inference(model, data, args)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--intel_extension_for_pytorch", default=False, action="store_true")
  parser.add_argument("--num_streams", default=1, type=int)

  main(parser.parse_args())


## python3 -m intel_extension_for_pytorch.cpu.launch --ninstance 1 --node_id 0 resnet50.py