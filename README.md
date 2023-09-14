# Latency Server NVIDIA Jetson AGX Orin

This README shows how to run latency measurements on NVIDIA Jetson AGX Orin.

Measurement server is based on:

- `trtexec` — standard TensorRT component that can measure inference time,
- [ENOT Latency Server](https://enot-autodl.rtd.enot.ai/en/latest/latency_server.html) — small open-source package that provides simple API for latency measurement.

The repository code was tested on Python 3.8.

To install the required packages run the following command:

```bash
pip install -r requirements.txt
```

Run a measurement server on Jetson:

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

> **_NOTE:_** If you pass a model with `QuantizeLinear` and `DequantizeLinear` layers to latency server, an engine with INT8 kernels will be automatically created.

We get stable results with the following parameter values (default values for our measurements):

- `warmUp`: `10000` (10 sec)
- `iterations`: `10000`
- `avgRuns`: `100`

Parameter values can be checked by the following command:

```bash
python tools/server.py --help
```

To measure latency, use the following command:

```bash
python tools/measure.py --model-onnx=model.onnx
```

If you are running the client (`tools/measure.py` script) on another computer, please install the necessary packages first and then specify the server address using `--host` and `--port` arguments.

⚠️ Summary:

- run `tools/server.py` on a target device (NVIDIA Jetson AGX Orin),
- run `tools/measure.py` with the specified server address.

## Unstable engine building

TensorRT sometimes builds an FP32 engine even if we pass `--fp16` flag to `trtexec`, this affect the measurement results ([issue](https://github.com/NVIDIA/TensorRT/issues/3160)).

To make sure that the engine is correct, we compare its size with the reference size: FP32 engine size or ONNX model size if `--compare-with-onnx` is passed.
If the size of the built engine is too large, then it is incorrect, and we automatically rebuild it.

The measurement script uses `1.5` as a default threshold on `reference size / current engine size` value (this value can be changed using `--threshold` option).
Latency server tries to build a correct engine for `--n-trials` times (20 by default) until `reference size / current engine size` becomes higher than the threshold.

If `trtexec` has failed to create a correct engine for `n_trials` times, latency server returns `None` as model latency.
If you want to know the actual `reference size / current engine size` ratio, use `--verbosity-level=1`.
