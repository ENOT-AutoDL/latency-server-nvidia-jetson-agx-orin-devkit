# Latency Server NVIDIA Jetson AGX Orin

This README shows how to run latency measurements on NVIDIA Jetson AGX Orin.

Measurement server is based on:

- `trtexec` — standard TensorRT component that can measure inference time,
- [ENOT Latency Server](https://enot-autodl.rtd.enot.ai/en/latest/latency_server.html)
  — small open-source package that provides simple API for latency measurement.

Repository code was tested on Python 3.8.

To install required packages run the following command:

```bash
pip install -r requirements.txt
```

Run measurement server on Jetson:

```bash
python tools/server.py
```

The server gets a model in the ONNX format and measures its latency using `trtexec`:

```bash
<trtexec_path> \
    --onnx=<onnx_model_path> \
    --warmUp=<warmup> \
    --iterations=<iterations> \
    --avgRuns=<avgruns> \
    --noDataTransfers \
    --useSpinWait \
    --useCudaGraph \
    --separateProfileRun \
    --percentile=95 \
    --fp16
```

We get stable results with the following parameter values (default values for our measurements):

- `warmUp`: `10000` (10 sec)
- `iterations`: `10000`
- `avgRuns`: `100`

Parameter values can be checked/changed by the following command:

```bash
python tools/server.py --help
```

To measure latency, use the following command:

```bash
python tools/measure.py --model-onnx=<onnx_model_path>
```

If you are running the client (`tools/measure.py` script) on another computer,
please install the necessary packages first
and then specify the server address using `--host` and `--port` arguments.

⚠️ Summary:

- run `tools/server.py` on a target device (NVIDIA AGX Jetson Orin),
- run `tools/measure.py` with the specified server address.

### Unstable engine building

TensorRT sometimes builds an FP32 engine even if we pass `--fp16` flag to `trtexec`,
this affects the measurement results ([issue](https://github.com/NVIDIA/TensorRT/issues/3160)).

To make sure that the engine is correct, we compare its size with the reference size:
FP32 engine size or ONNX size if `--compare-with-onnx` is passed.
If the size of the built engine is too large, then it is incorrect and we automatically rebuild it.

The measurement server uses `1.9` as a threshold on `reference_size / current_engine_size` value.
New engines will be generated until `reference_size / current_engine_size` becomes higher than the threshold. 
This value can be changed using `--threshold` option.
If you want to know the actual size ratio, please use `--verbose=1` argument.
