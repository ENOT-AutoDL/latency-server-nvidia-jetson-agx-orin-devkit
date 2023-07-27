import argparse

import onnx
from enot_latency_server.client import measure_latency_remote

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-onnx", required=True, help="Path to model onnx for latency measurement")
    parser.add_argument("--host", default="localhost", type=str, help="Host of latency measurement server")
    parser.add_argument("--port", default=15003, type=int, help="Port of latency measurement server")
    args = parser.parse_args()

    onnx_model = onnx.load(args.model_onnx)

    print("please wait (average time for 1 measurement 5 minutes)")
    result = measure_latency_remote(
        onnx_model.SerializeToString(),
        host=args.host,
        port=args.port,
    )

    print(result)
