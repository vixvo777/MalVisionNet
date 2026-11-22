"""
Export trained models to ONNX format with optional quantization.
"""
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from ..logger import logger

def export_to_onnx(model, config_path: str, opset_version=17):
    """
    Export PyTorch model to ONNX format.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set model to evaluation mode
    model.eval()

    # Create dummy input based on representation
    if config['representation'] == 'grayscale':
        dummy_input = torch.randn(1, 1, *config['resize_target'])
    else:
        dummy_input = torch.randn(1, 3, *config['resize_target'])

    # Export to ONNX
    onnx_path = Path(config['final_model_path']).with_suffix('.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    logger.info(f"ONNX model saved to {onnx_path}")
    return onnx_path

def quantize_onnx(onnx_path: Path, calibration_data=None):
    """
    Quantize ONNX model to INT8 using ONNX Runtime.
    """
    # Determine quantization type
    if calibration_data is None:
        quantization_type = 'dynamic'
    else:
        quantization_type = 'static'
        # For static quantization, calibration data must be provided
        # This would typically be a representative dataset
        # Placeholder: In practice, use a subset of training data
        pass

    # Load ONNX model
    sess = ort.InferenceSession(str(onnx_path))

    # Create quantized model
    quantized_path = onnx_path.with_suffix('.quant.onnx')

    # For static quantization, use calibration data
    # This is a simplified example; full implementation requires calibration
    if quantization_type == 'static':
        logger.warning("Static quantization requires calibration data (not implemented in this example)")
        return onnx_path

    # Perform dynamic quantization
    from onnxruntime.quantization import quantize_dynamic,
    from onnxruntime.quantization.quant_utils import get_op_types_to_quantize

    op_types_to_quantize = get_op_types_to_quantize(quantization_type)
    quantize_dynamic(
        onnx_path,
        quantized_path,
        op_types_to_quantize=op_types_to_quantize,
        per_channel=False
    )
    logger.info(f"Quantized model saved to {quantized_path}")
    return quantized_path


def benchmark_onnx(onnx_path: Path, batch_size=1):
    """
    Benchmark ONNX model latency.
    """
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    # Create dummy input
    if 'grayscale' in onnx_path.as_posix():
        dummy_input = np.random.rand(batch_size, 1, 224, 224).astype(np.float32)
    else:
        dummy_input = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)

    # Warm-up
    for _ in range(10):
        sess.run([], {input_name: dummy_input})

    # Benchmark
    import time
    iterations = 100
    total_time = 0
    for _ in range(iterations):
        start = time.time()
        sess.run([], {input_name: dummy_input})
        total_time += time.time() - start

    avg_latency = total_time / iterations
    throughput = batch_size / avg_latency

    logger.info(f"ONNX Benchmark ({batch_size} batch):")
    logger.info(f"  Avg Latency: {avg_latency*1000:.2f} ms")
    logger.info(f"  Throughput: {throughput:.2f} samples/sec")

    return avg_latency

if __name__ == "__main__":
    # Example usage
    # model = SimpleResNet18()
    # onnx_path = export_to_onnx(model, "configs/train.yaml")
    # quantize_onnx(onnx_path)
    pass