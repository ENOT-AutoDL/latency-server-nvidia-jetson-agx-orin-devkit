import argparse
import os
import subprocess
from typing import Dict

import aiohttp
import onnx
from enot_latency_server.server import LatencyServer


class MyServer(LatencyServer):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 15003,
        trtexec_path: str = "/usr/src/tensorrt/bin/trtexec",
        fp32: bool = False,
        warmup: int = 10000,
        iterations: int = 10000,
        avgruns: int = 100,
        compare_with_onnx: bool = False,
        verbosity_level: int = 0,
        threshold: float = 1.5,
        n_trials: int = 20,
    ):
        """
        Server ctor.

        Parameters
        ----------
        host : str
            Host name or IP address. Default value is ``0.0.0.0``.
        port : int
            Port. Default value is 15003.
        trtexec_path : str
            Path to trtexec binaries
        fp32 : bool
            Whether to use an FP32 engine for inference instead of an FP16 one.
            Default is False.
        warmup : int
            Run for N milliseconds to warmup before measuring performance.
            Default is 10000.
        iterations : int
            Run at least ``iterations`` inference iterations for latency measurement
            Default is 10000.
        avgruns : int
            Report performance measurements averaged over ``avgruns`` consecutive iterations
            Default is 100.
        compare_with_onnx : bool
            Whether to compare FP16 engine size with ONNX model size. If false, compares with FP32 engine size.
            Not used when ``fp32`` is False.
            Default is False.
        threshold : float
            Ratio of reference size (i.e. ONNX model size) to engine size to make sure we have an FP16 engine.
            Not used when ``fp32`` is False.
            Default is 1.5
        verbosity_level : int
            Verbosity level.
            Choices: 0 - stderr, 1 - show measurement results, 2 - both stderr and measurement results.
            Default is 0.
        n_trials : int
            Try to build an FP16 engine for ``n_trials`` times before returning ``None``.
            Not used when ``fp32 == False``.
            Default is 20.

        """
        super().__init__(host=host, port=port)
        self.trtexec_path = trtexec_path

        self._fp32 = fp32
        self._fp16 = not self._fp32
        self.warmup = warmup
        self.iterations = iterations
        self.avgruns = avgruns
        self.compare_with_onnx = compare_with_onnx
        self.verbosity_level = verbosity_level
        self.threshold = threshold

        if n_trials < 1:
            raise ValueError(f"`n_trials` should be a positive number, got {n_trials}")

        self.n_trials = n_trials

    @staticmethod
    def floats_from_str(str_data):
        floats = [float(x) for x in str_data.split(" ") if x.replace(".", "").isdigit()]
        return floats

    def parse_trtexec_stdout(self, process_stdout):
        summary_start = process_stdout.find("=== Performance summary ===")
        summary = process_stdout[summary_start:]
        start_ind = summary.find("Latency")
        stop_ind = summary.find("Enqueue Time")

        summary_content = summary[start_ind:stop_ind]
        latency_data = self.floats_from_str(summary_content)

        if len(latency_data) < 5:  # len(latency_data) should be 5 for TensorRT.trtexec [TensorRT v8401]
            raise aiohttp.web.HTTPInternalServerError(
                reason=f"Something went wrong, cannot parse trtexec output! Summary content: {summary_content}"
            )

        title = ("t_min", "t_max", "t_mean", "t_median", "percentile(95%)")
        result = {name: x for name, x in zip(title, latency_data)}
        result["latency"] = result["t_mean"]

        return result

    def get_engine_size(self, process_stdout, eps=1e-7):
        start_ind = process_stdout.find("engine size:")
        stop_ind = process_stdout[start_ind:].find("MiB")

        engine_data = process_stdout[start_ind : start_ind + stop_ind + 3].strip()
        floats = self.floats_from_str(engine_data)
        if len(floats) != 1:
            raise aiohttp.web.HTTPInternalServerError(
                reason=f"Something went wrong, cannot parse trtexec output! Engine data: {engine_data}"
            )

        engine_size = floats[0]
        return engine_size + eps

    def run_subprocess(self, command):
        print(f"Running `{command}`")
        pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True)
        process_stdout, process_stderr = pipe.stdout.decode("utf-8"), pipe.stderr.decode("utf-8")

        if self.verbosity_level in (0, 2):  # show stderr
            print(process_stderr)
        return process_stdout, process_stderr

    def measure_latency(
        self,
        model: bytes,
    ) -> Dict[str, float]:
        """
        Method that allows to measure latency using TensorRT framework.

        Parameters
        ----------
        model : bytes
            Model which latency we want to measure.
        Returns
        -------
        Dict[str, float]
            dict with measured parameters

        """
        model = onnx.load_model_from_string(model)
        onnx.save(model, "model.onnx")

        engine_options = "--noDataTransfers --useSpinWait --useCudaGraph --separateProfileRun --percentile=95"
        base_command = f"{self.trtexec_path} --onnx=model.onnx {engine_options}"
        run_options = f"--warmUp={self.warmup} --iterations={self.iterations} --avgRuns={self.avgruns}"

        if _has_quant_dequant_node(model):
            print("Found quant-dequant nodes, INT8 kernels usage is enabled.")

            if self._fp16:
                command_int8 = f"{base_command} --int8 --fp16 {run_options}"
            else:  # FP32
                command_int8 = f"{base_command} --int8 {run_options}"

            command_stdout, command_stderr = self.run_subprocess(command_int8)
        elif self._fp32:
            command_fp32 = f"{base_command} {run_options}"
            command_stdout, command_stderr = self.run_subprocess(command_fp32)
        else:  # FP16
            command_fp32 = base_command  # with default run options because we only need engine size
            command_fp16 = f"{base_command} --fp16 {run_options}"

            if self.compare_with_onnx:
                onnx_size = os.path.getsize("model.onnx") / 1024**2  # in MiB
                reference_size = onnx_size
            else:  # compare with engine
                stdout_fp32, stderr_fp32 = self.run_subprocess(command_fp32)

                fp32_engine_size = self.get_engine_size(stdout_fp32)
                reference_size = fp32_engine_size

            # Sometimes trtexec creates engine in FP32 even though we demand FP16.
            # To make sure we get FP16 engine, we compare its size with reference size.
            is_fp16_engine = False

            for n_trial in range(self.n_trials):  # Rebuild an engine if it is too large.
                command_stdout, command_stderr = self.run_subprocess(command_fp16)
                engine_size = self.get_engine_size(command_stdout)
                engine_ratio = reference_size / engine_size

                if self.verbosity_level in (1, 2):
                    print(
                        f"Trial {n_trial + 1}, Reference_size / engine_size = {engine_ratio}, threshold = {self.threshold}"
                    )

                if engine_ratio > self.threshold:
                    is_fp16_engine = True
                    break

        if self._fp16 and not is_fp16_engine:
            result = {"latency": None}
        else:
            result = self.parse_trtexec_stdout(command_stdout)

        if self.verbosity_level in (1, 2):  # show measurements result
            print("=" * 10, " Results ", "=" * 10)
            print(result)

        return result


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trtexec-path", type=str, default="/usr/src/tensorrt/bin/trtexec", help="Path to trtexec binaries."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host name or IP address. Default value is `0.0.0.0`."
    )
    parser.add_argument("--port", type=int, default=15003, help="Server port. Default is 15003")
    parser.add_argument(
        "--fp32", action="store_true", help="Whether to build an FP32 engine. Builds FP16 engine by default."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10000,
        help="Run for `warmup` milliseconds to warmup before measuring performance. Default is 10000.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Run at least `iterations` inference iterations for latency measurement. Default is 10000.",
    )
    parser.add_argument(
        "--avgruns",
        type=int,
        default=100,
        help="Report performance measurements averaged over `avgruns` consecutive iterations. Default is 100.",
    )
    parser.add_argument(
        "--compare-with-onnx",
        action="store_true",
        help="Use ONNX model size as a reference to compare with FP16 engine size. "
        "By default, uses FP32 engine size as a reference. Not used with `--fp32`.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Ratio of reference size (i.e. ONNX model size) to engine size to make sure we have an FP16 engine. "
        "Not used with `--fp32`. Default is 1.5.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Try to build an FP16 engine for `n_trials` times before returning `None`. "
        "Not used with `--fp32`. Default is 20.",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Choices: 0 - stderr, 1 - show measurement results, 2 - both stderr and measurement results. "
        "Default is 0.",
    )

    return parser.parse_args()


def main():
    args = parse()
    print(args)
    server = MyServer(
        host=args.host,
        port=args.port,
        trtexec_path=args.trtexec_path,
        fp32=args.fp32,
        warmup=args.warmup,
        iterations=args.iterations,
        avgruns=args.avgruns,
        compare_with_onnx=args.compare_with_onnx,
        verbosity_level=args.verbosity_level,
        threshold=args.threshold,
        n_trials=args.n_trials,
    )
    server.run()


def _has_quant_dequant_node(onnx_model: onnx.ModelProto) -> bool:
    """Checks if ONNX model has quant-dequant nodes."""
    for node in onnx_model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            return True

    return False


if __name__ == "__main__":
    main()
