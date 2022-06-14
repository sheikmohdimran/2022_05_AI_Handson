### Steps to convert YoloV3 to OpenVINO format.

Create conda environment for YOLO
```bash
conda create -n yolo -y 
conda activate yolo
```

Install required libraries
```bash
pip install tidecv tfcoreml imagecorruptions pycocotools onnx mnn keras2onnx tf2onnx onnxruntime opencv-contrib-python tensorflow-model-optimization keras_applications imgaug 

conda install numpy scipy scikit-learn tensorflow matplotlib tqdm pillow Cython sympy bokeh -y

```

Activate conda and download weights
```bash
conda activate yolo
mkdir files
export dl_dir=`pwd`/files
cd $dl_dir

git clone https://github.com/david8862/keras-YOLOv3-model-set.git
cd keras-YOLOv3-model-set

wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
```

Covert yolov3 weights to h5 format

```bash
python tools/model_converter/convert.py cfg/yolov3.cfg weights/yolov3.weights weights/yolov3.h5
```

Convert h5 weights to pb format
```bash
pip install opencv-python keras_applications
python tools/model_converter/keras_to_tensorflow.py --input_model weights/yolov3.h5 --output_model=weights/yolo-v3.pb
```

Start up the docker container
```bash
docker run -u 0 -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --privileged -v  /home/sdp/Documents/files:/files openvino/ubuntu18_dev
```

Install required libraries to compile the C++ compilation
```bash
bash
apt update
apt install cmake gcc g++ numactl -y
```

Build the predefined samples
```bash
cd samples/cpp
./build_samples.sh
export OV_DIR=/root/inference_engine_cpp_samples_build/intel64/Release
```

Convert the pb model to OpenVINO IR format
```bash
mo --input_model /files/keras-YOLOv3-model-set/weights/yolo-v3.pb --input_shape=[1,416,416,3] 
```


Get benchmarks for different combinations
```bash
numactl -C 0,1 -m 0 -N 0 benchmark_app -m yolo-v3.xml -niter 10 -nireq 4 -nstreams 2 -b 1 -nthreads 2

numactl -C 0,1 -m 0 -N 0 benchmark_app -m yolo-v3.xml -niter 10 -nireq 4 -nstreams 2 -b 2 -nthreads 2

numactl -C 0,1 -m 0 -N 0 benchmark_app -m yolo-v3.xml -niter 10 -nireq 4 -nstreams 2 -b 3 -nthreads 2
```