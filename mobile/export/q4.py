import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
model = onnx.load("flow_lm_main_kv.onnx")
cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4)
q = MatMulNBitsQuantizer(model, algo_config=cfg)
q.process()
q.model.save_model_to_file("flow_lm_main_kv_q4.onnx", use_external_data_format=False)
print("done")
