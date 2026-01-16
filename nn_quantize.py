"""
Post-Training Quantization for NNUE/DNN Networks

This module provides quantization functionality to convert float32 weights to int8/int16
for faster inference and smaller model size.

Quantization Scheme:
- Feature Transform (ft) layer: int16 weights, int32 accumulator
  (Larger range needed because accumulator sums ~30 feature weights)
- Hidden layers (l1, l2, l3): int8 weights
- Activations: Clipped ReLU [0, 1] maps to [0, 127] for int8

Scale Factors:
- Weights are scaled to fit in quantized range while preserving precision
- Activations use fixed-point representation: value = int_value / 127

Usage:
    # Quantize a trained model
    from nn_quantize import quantize_model, QuantizedNNUEInference

    quantized = quantize_model(model, "NNUE")
    quantized.save("model_quantized.bin")

    # Load and use quantized model
    q_inference = QuantizedNNUEInference.load("model_quantized.bin")
    score = q_inference.evaluate_board(board)

    # Compare with float inference
    from nn_inference import NNUEInference
    float_inference = NNUEInference(model)
    score_float = float_inference.evaluate_board(board)
"""

import struct
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import chess

from nn_inference import (
    NNUENetwork, DNNNetwork, NNUEFeatures, DNNFeatures,
    NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE, DNN_INPUT_SIZE, DNN_HIDDEN_LAYERS,
    TANH_SCALE
)

# Quantization constants
FT_WEIGHT_SCALE = 127  # Feature transformer weights scaled to int16
FT_ACTIVATION_SCALE = 127  # Accumulator output scale (after clipped ReLU)
HIDDEN_WEIGHT_SCALE = 64  # Hidden layer weights scaled to int8
HIDDEN_ACTIVATION_SCALE = 127  # Hidden layer activation scale [0, 127] for [0, 1]
OUTPUT_SCALE = 127 * 64  # Combined scale for output layer


class QuantizedWeights:
    """Container for quantized weights with scale factors"""

    def __init__(self):
        # Feature transformer (int16)
        self.ft_weight: Optional[np.ndarray] = None  # int16
        self.ft_bias: Optional[np.ndarray] = None  # int32
        self.ft_weight_scale: float = 1.0

        # Hidden layers (int8)
        self.l1_weight: Optional[np.ndarray] = None  # int8
        self.l1_bias: Optional[np.ndarray] = None  # int32
        self.l2_weight: Optional[np.ndarray] = None  # int8
        self.l2_bias: Optional[np.ndarray] = None  # int32
        self.l3_weight: Optional[np.ndarray] = None  # int8
        self.l3_bias: Optional[np.ndarray] = None  # int32

        # DNN-specific (l4)
        self.l4_weight: Optional[np.ndarray] = None  # int8
        self.l4_bias: Optional[np.ndarray] = None  # int32

        # Network metadata
        self.nn_type: str = "NNUE"
        self.input_size: int = NNUE_INPUT_SIZE
        self.hidden_size: int = NNUE_HIDDEN_SIZE

        # Scale factors for each layer
        self.scales: Dict[str, float] = {}

    def save(self, filepath: str):
        """Save quantized weights to binary file"""
        with open(filepath, 'wb') as f:
            # Header: magic, version, nn_type
            f.write(b'QNNUE')  # Magic bytes
            f.write(struct.pack('I', 1))  # Version
            f.write(struct.pack('I', 1 if self.nn_type == "NNUE" else 2))
            f.write(struct.pack('I', self.input_size))
            f.write(struct.pack('I', self.hidden_size))

            # Number of scale factors
            f.write(struct.pack('I', len(self.scales)))
            for name, scale in self.scales.items():
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('f', scale))

            # Feature transformer (int16 weights, int32 bias)
            self._write_array_int16(f, self.ft_weight)
            self._write_array_int32(f, self.ft_bias)

            # Hidden layers (int8 weights, int32 bias)
            self._write_array_int8(f, self.l1_weight)
            self._write_array_int32(f, self.l1_bias)
            self._write_array_int8(f, self.l2_weight)
            self._write_array_int32(f, self.l2_bias)
            self._write_array_int8(f, self.l3_weight)
            self._write_array_int32(f, self.l3_bias)

            # DNN-specific l4 layer
            if self.nn_type == "DNN":
                self._write_array_int8(f, self.l4_weight)
                self._write_array_int32(f, self.l4_bias)

        print(f"Saved quantized weights to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'QuantizedWeights':
        """Load quantized weights from binary file"""
        qw = cls()

        with open(filepath, 'rb') as f:
            # Header
            magic = f.read(5)
            if magic != b'QNNUE':
                raise ValueError(f"Invalid file format: {magic}")

            version = struct.unpack('I', f.read(4))[0]
            nn_type_code = struct.unpack('I', f.read(4))[0]
            qw.nn_type = "NNUE" if nn_type_code == 1 else "DNN"
            qw.input_size = struct.unpack('I', f.read(4))[0]
            qw.hidden_size = struct.unpack('I', f.read(4))[0]

            # Scale factors
            num_scales = struct.unpack('I', f.read(4))[0]
            for _ in range(num_scales):
                name_len = struct.unpack('I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                scale = struct.unpack('f', f.read(4))[0]
                qw.scales[name] = scale

            # Feature transformer
            qw.ft_weight = qw._read_array_int16(f)
            qw.ft_bias = qw._read_array_int32(f)

            # Hidden layers
            qw.l1_weight = qw._read_array_int8(f)
            qw.l1_bias = qw._read_array_int32(f)
            qw.l2_weight = qw._read_array_int8(f)
            qw.l2_bias = qw._read_array_int32(f)
            qw.l3_weight = qw._read_array_int8(f)
            qw.l3_bias = qw._read_array_int32(f)

            # DNN-specific
            if qw.nn_type == "DNN":
                qw.l4_weight = qw._read_array_int8(f)
                qw.l4_bias = qw._read_array_int32(f)

        return qw

    def _write_array_int8(self, f, arr: np.ndarray):
        """Write int8 array with shape info"""
        if arr is None:
            f.write(struct.pack('I', 0))
            return
        shape = arr.shape
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
        f.write(arr.astype(np.int8).tobytes())

    def _write_array_int16(self, f, arr: np.ndarray):
        """Write int16 array with shape info"""
        if arr is None:
            f.write(struct.pack('I', 0))
            return
        shape = arr.shape
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
        f.write(arr.astype(np.int16).tobytes())

    def _write_array_int32(self, f, arr: np.ndarray):
        """Write int32 array with shape info"""
        if arr is None:
            f.write(struct.pack('I', 0))
            return
        shape = arr.shape
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
        f.write(arr.astype(np.int32).tobytes())

    def _read_array_int8(self, f) -> Optional[np.ndarray]:
        """Read int8 array"""
        ndims = struct.unpack('I', f.read(4))[0]
        if ndims == 0:
            return None
        shape = tuple(struct.unpack('I', f.read(4))[0] for _ in range(ndims))
        size = np.prod(shape)
        data = np.frombuffer(f.read(size), dtype=np.int8)
        return data.reshape(shape)

    def _read_array_int16(self, f) -> Optional[np.ndarray]:
        """Read int16 array"""
        ndims = struct.unpack('I', f.read(4))[0]
        if ndims == 0:
            return None
        shape = tuple(struct.unpack('I', f.read(4))[0] for _ in range(ndims))
        size = np.prod(shape) * 2
        data = np.frombuffer(f.read(size), dtype=np.int16)
        return data.reshape(shape)

    def _read_array_int32(self, f) -> Optional[np.ndarray]:
        """Read int32 array"""
        ndims = struct.unpack('I', f.read(4))[0]
        if ndims == 0:
            return None
        shape = tuple(struct.unpack('I', f.read(4))[0] for _ in range(ndims))
        size = np.prod(shape) * 4
        data = np.frombuffer(f.read(size), dtype=np.int32)
        return data.reshape(shape)


def quantize_nnue(model: NNUENetwork) -> QuantizedWeights:
    """
    Quantize NNUE model weights to int8/int16.

    Quantization strategy:
    - Feature transformer (ft): int16 weights to preserve precision
      (accumulator sums ~30 weights, needs larger range)
    - Hidden layers (l1, l2, l3): int8 weights
    - Biases: int32 to match accumulator precision

    Scale factors are computed per-layer based on weight ranges.
    """
    qw = QuantizedWeights()
    qw.nn_type = "NNUE"
    qw.input_size = model.input_size
    qw.hidden_size = model.hidden_size

    model.eval()
    with torch.no_grad():
        # Feature transformer - use int16 for better precision
        ft_weight = model.ft.weight.cpu().numpy()  # (hidden_size, input_size)
        ft_bias = model.ft.bias.cpu().numpy()  # (hidden_size,)

        # Compute scale for ft weights
        ft_max = max(np.abs(ft_weight).max(), 1e-8)
        ft_scale = 32767.0 / ft_max  # Scale to fit int16 range
        qw.scales['ft_weight'] = ft_scale

        # Quantize ft weights to int16
        qw.ft_weight = np.round(ft_weight * ft_scale).astype(np.int16)

        # Bias is scaled to match: accumulator = sum(w * x) where x is 0/1
        # After dequant: accumulator_float = accumulator_int / ft_scale
        # So bias should be: bias_int = bias_float * ft_scale
        qw.ft_bias = np.round(ft_bias * ft_scale).astype(np.int32)

        # Hidden layers use int8
        # After clipped ReLU, activations are in [0, 1], we scale to [0, 127]
        activation_scale = 127.0
        qw.scales['activation'] = activation_scale

        # L1: input from concatenated accumulators (512 values in [0, 1])
        l1_weight = model.l1.weight.cpu().numpy()  # (32, 512)
        l1_bias = model.l1.bias.cpu().numpy()  # (32,)

        l1_max = max(np.abs(l1_weight).max(), 1e-8)
        l1_scale = 127.0 / l1_max
        qw.scales['l1_weight'] = l1_scale

        qw.l1_weight = np.round(l1_weight * l1_scale).astype(np.int8)
        # Bias scale: input is scaled by activation_scale, weights by l1_scale
        # output = (input/127) * (weight/l1_scale) + bias
        # To match: bias_int / (127 * l1_scale) = bias_float
        # So: bias_int = bias_float * 127 * l1_scale
        qw.l1_bias = np.round(l1_bias * activation_scale * l1_scale).astype(np.int32)

        # L2
        l2_weight = model.l2.weight.cpu().numpy()
        l2_bias = model.l2.bias.cpu().numpy()

        l2_max = max(np.abs(l2_weight).max(), 1e-8)
        l2_scale = 127.0 / l2_max
        qw.scales['l2_weight'] = l2_scale

        qw.l2_weight = np.round(l2_weight * l2_scale).astype(np.int8)
        qw.l2_bias = np.round(l2_bias * activation_scale * l2_scale).astype(np.int32)

        # L3 (output layer - no activation)
        l3_weight = model.l3.weight.cpu().numpy()
        l3_bias = model.l3.bias.cpu().numpy()

        l3_max = max(np.abs(l3_weight).max(), 1e-8)
        l3_scale = 127.0 / l3_max
        qw.scales['l3_weight'] = l3_scale

        qw.l3_weight = np.round(l3_weight * l3_scale).astype(np.int8)
        qw.l3_bias = np.round(l3_bias * activation_scale * l3_scale).astype(np.int32)

    return qw


def quantize_dnn(model: DNNNetwork) -> QuantizedWeights:
    """
    Quantize DNN model weights to int8/int16.
    Similar to NNUE but with 4 layers instead of 3 hidden.
    """
    qw = QuantizedWeights()
    qw.nn_type = "DNN"
    qw.input_size = model.input_size
    qw.hidden_size = model.hidden_layers[0]

    model.eval()
    with torch.no_grad():
        activation_scale = 127.0
        qw.scales['activation'] = activation_scale

        # L1 (first layer - like feature transformer)
        l1_weight = model.l1.weight.cpu().numpy()
        l1_bias = model.l1.bias.cpu().numpy()

        l1_max = max(np.abs(l1_weight).max(), 1e-8)
        l1_scale = 32767.0 / l1_max  # int16 for first layer
        qw.scales['ft_weight'] = l1_scale  # Use same name for consistency

        qw.ft_weight = np.round(l1_weight * l1_scale).astype(np.int16)
        qw.ft_bias = np.round(l1_bias * l1_scale).astype(np.int32)

        # L2
        l2_weight = model.l2.weight.cpu().numpy()
        l2_bias = model.l2.bias.cpu().numpy()

        l2_max = max(np.abs(l2_weight).max(), 1e-8)
        l2_scale = 127.0 / l2_max
        qw.scales['l1_weight'] = l2_scale  # Offset naming by 1

        qw.l1_weight = np.round(l2_weight * l2_scale).astype(np.int8)
        qw.l1_bias = np.round(l2_bias * activation_scale * l2_scale).astype(np.int32)

        # L3
        l3_weight = model.l3.weight.cpu().numpy()
        l3_bias = model.l3.bias.cpu().numpy()

        l3_max = max(np.abs(l3_weight).max(), 1e-8)
        l3_scale = 127.0 / l3_max
        qw.scales['l2_weight'] = l3_scale

        qw.l2_weight = np.round(l3_weight * l3_scale).astype(np.int8)
        qw.l2_bias = np.round(l3_bias * activation_scale * l3_scale).astype(np.int32)

        # L4 (output)
        l4_weight = model.l4.weight.cpu().numpy()
        l4_bias = model.l4.bias.cpu().numpy()

        l4_max = max(np.abs(l4_weight).max(), 1e-8)
        l4_scale = 127.0 / l4_max
        qw.scales['l3_weight'] = l4_scale

        qw.l3_weight = np.round(l4_weight * l4_scale).astype(np.int8)
        qw.l3_bias = np.round(l4_bias * activation_scale * l4_scale).astype(np.int32)

    return qw


def quantize_model(model: torch.nn.Module, nn_type: str) -> QuantizedWeights:
    """
    Quantize a trained model.

    Args:
        model: Trained PyTorch model (NNUENetwork or DNNNetwork)
        nn_type: "NNUE" or "DNN"

    Returns:
        QuantizedWeights object containing int8/int16 weights
    """
    if nn_type == "NNUE":
        return quantize_nnue(model)
    elif nn_type == "DNN":
        return quantize_dnn(model)
    else:
        raise ValueError(f"Unknown nn_type: {nn_type}")


class QuantizedNNUEInference:
    """
    Quantized inference engine for NNUE.
    Uses integer arithmetic for the forward pass.
    """

    def __init__(self, qw: QuantizedWeights):
        """Initialize from quantized weights"""
        if qw.nn_type != "NNUE":
            raise ValueError("QuantizedNNUEInference requires NNUE weights")

        self.qw = qw
        self.hidden_size = qw.hidden_size

        # Precompute combined scales for output conversion
        ft_scale = qw.scales['ft_weight']
        activation_scale = qw.scales['activation']
        l1_scale = qw.scales['l1_weight']
        l2_scale = qw.scales['l2_weight']
        l3_scale = qw.scales['l3_weight']

        # Scale to convert final int32 output back to float
        # Each layer introduces: output_int = input_int * weight_int / scale
        # We need to track cumulative scaling
        self.ft_scale = ft_scale
        self.l1_combined_scale = ft_scale * l1_scale / activation_scale
        self.l2_combined_scale = self.l1_combined_scale * l2_scale / activation_scale
        self.l3_combined_scale = self.l2_combined_scale * l3_scale / activation_scale

        # For accumulators
        self.white_accumulator: Optional[np.ndarray] = None
        self.black_accumulator: Optional[np.ndarray] = None

    @classmethod
    def load(cls, filepath: str) -> 'QuantizedNNUEInference':
        """Load quantized model from file"""
        qw = QuantizedWeights.load(filepath)
        return cls(qw)

    def _clipped_relu_int(self, x: np.ndarray, scale: float) -> np.ndarray:
        """
        Clipped ReLU for integer values.
        Maps [0, scale] to [0, 127] (int8 range for activations).
        """
        # Clip to [0, scale] then normalize to [0, 127]
        x = np.clip(x, 0, scale)
        return (x * 127 / scale).astype(np.int32)

    def evaluate_full(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        """
        Full quantized evaluation.

        Args:
            white_features: Active feature indices for white's perspective
            black_features: Active feature indices for black's perspective
            stm: True if white to move, False if black to move

        Returns:
            Evaluation score (float, approximately in [-1, 1])
        """
        # Feature transformer - accumulate int16 weights into int32
        white_acc = self.qw.ft_bias.copy().astype(np.int32)
        black_acc = self.qw.ft_bias.copy().astype(np.int32)

        ft_weight = self.qw.ft_weight  # (hidden_size, input_size) int16

        for f in white_features:
            if 0 <= f < ft_weight.shape[1]:
                white_acc += ft_weight[:, f].astype(np.int32)

        for f in black_features:
            if 0 <= f < ft_weight.shape[1]:
                black_acc += ft_weight[:, f].astype(np.int32)

        # Clipped ReLU: convert to [0, 127] range
        white_hidden = self._clipped_relu_int(white_acc, self.ft_scale)
        black_hidden = self._clipped_relu_int(black_acc, self.ft_scale)

        # Perspective concatenation
        if stm:
            hidden = np.concatenate([white_hidden, black_hidden])
        else:
            hidden = np.concatenate([black_hidden, white_hidden])

        # L1: int8 weights, int32 accumulator
        # Note: numpy will auto-promote int8 * int32 to int32/int64
        l1_out = np.dot(hidden, self.qw.l1_weight.T.astype(np.int32)) + self.qw.l1_bias
        l1_out = self._clipped_relu_int(l1_out, self.l1_combined_scale)

        # L2
        l2_out = np.dot(l1_out, self.qw.l2_weight.T.astype(np.int32)) + self.qw.l2_bias
        l2_out = self._clipped_relu_int(l2_out, self.l2_combined_scale)

        # L3 (output - no activation)
        l3_out = np.dot(l2_out, self.qw.l3_weight.T.astype(np.int32)) + self.qw.l3_bias

        # Convert back to float
        output = l3_out[0] / self.l3_combined_scale

        return float(output)

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a chess board position"""
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        return self.evaluate_full(white_feat, black_feat, board.turn == chess.WHITE)

    def _refresh_accumulator(self, white_features: List[int], black_features: List[int]):
        """Initialize accumulators for incremental evaluation"""
        self.white_accumulator = self.qw.ft_bias.copy().astype(np.int32)
        self.black_accumulator = self.qw.ft_bias.copy().astype(np.int32)

        ft_weight = self.qw.ft_weight

        for f in white_features:
            if 0 <= f < ft_weight.shape[1]:
                self.white_accumulator += ft_weight[:, f].astype(np.int32)

        for f in black_features:
            if 0 <= f < ft_weight.shape[1]:
                self.black_accumulator += ft_weight[:, f].astype(np.int32)

    def update_accumulator(self, added_white: set, removed_white: set,
                           added_black: set, removed_black: set):
        """Update accumulators incrementally"""
        if self.white_accumulator is None:
            raise RuntimeError("Accumulators not initialized")

        ft_weight = self.qw.ft_weight

        for f in added_white:
            if 0 <= f < ft_weight.shape[1]:
                self.white_accumulator += ft_weight[:, f].astype(np.int32)
        for f in removed_white:
            if 0 <= f < ft_weight.shape[1]:
                self.white_accumulator -= ft_weight[:, f].astype(np.int32)

        for f in added_black:
            if 0 <= f < ft_weight.shape[1]:
                self.black_accumulator += ft_weight[:, f].astype(np.int32)
        for f in removed_black:
            if 0 <= f < ft_weight.shape[1]:
                self.black_accumulator -= ft_weight[:, f].astype(np.int32)

    def evaluate_incremental(self, stm: bool) -> float:
        """Evaluate using current accumulators"""
        if self.white_accumulator is None:
            raise RuntimeError("Accumulators not initialized")

        white_hidden = self._clipped_relu_int(self.white_accumulator, self.ft_scale)
        black_hidden = self._clipped_relu_int(self.black_accumulator, self.ft_scale)

        if stm:
            hidden = np.concatenate([white_hidden, black_hidden])
        else:
            hidden = np.concatenate([black_hidden, white_hidden])

        l1_out = np.dot(hidden, self.qw.l1_weight.T.astype(np.int32)) + self.qw.l1_bias
        l1_out = self._clipped_relu_int(l1_out, self.l1_combined_scale)

        l2_out = np.dot(l1_out, self.qw.l2_weight.T.astype(np.int32)) + self.qw.l2_bias
        l2_out = self._clipped_relu_int(l2_out, self.l2_combined_scale)

        l3_out = np.dot(l2_out, self.qw.l3_weight.T.astype(np.int32)) + self.qw.l3_bias

        return float(l3_out[0] / self.l3_combined_scale)


class QuantizedDNNInference:
    """
    Quantized inference engine for DNN.
    Uses integer arithmetic for the forward pass.
    """

    def __init__(self, qw: QuantizedWeights):
        """Initialize from quantized weights"""
        if qw.nn_type != "DNN":
            raise ValueError("QuantizedDNNInference requires DNN weights")

        self.qw = qw

        # Precompute scales
        ft_scale = qw.scales['ft_weight']
        activation_scale = qw.scales['activation']
        l1_scale = qw.scales.get('l1_weight', 64.0)
        l2_scale = qw.scales.get('l2_weight', 64.0)
        l3_scale = qw.scales.get('l3_weight', 64.0)

        self.ft_scale = ft_scale
        self.l1_combined_scale = ft_scale * l1_scale / activation_scale
        self.l2_combined_scale = self.l1_combined_scale * l2_scale / activation_scale
        self.l3_combined_scale = self.l2_combined_scale * l3_scale / activation_scale

    @classmethod
    def load(cls, filepath: str) -> 'QuantizedDNNInference':
        """Load quantized model from file"""
        qw = QuantizedWeights.load(filepath)
        return cls(qw)

    def _clipped_relu_int(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Clipped ReLU for integer values"""
        x = np.clip(x, 0, scale)
        return (x * 127 / scale).astype(np.int32)

    def evaluate_full(self, features: List[int]) -> float:
        """Full quantized evaluation"""
        # First layer (feature transform)
        acc = self.qw.ft_bias.copy().astype(np.int32)
        ft_weight = self.qw.ft_weight

        for f in features:
            if 0 <= f < ft_weight.shape[1]:
                acc += ft_weight[:, f].astype(np.int32)

        x = self._clipped_relu_int(acc, self.ft_scale)

        # L2 (called l1 in qw due to offset)
        x = np.dot(x, self.qw.l1_weight.T.astype(np.int32)) + self.qw.l1_bias
        x = self._clipped_relu_int(x, self.l1_combined_scale)

        # L3
        x = np.dot(x, self.qw.l2_weight.T.astype(np.int32)) + self.qw.l2_bias
        x = self._clipped_relu_int(x, self.l2_combined_scale)

        # L4 (output)
        x = np.dot(x, self.qw.l3_weight.T.astype(np.int32)) + self.qw.l3_bias

        return float(x[0] / self.l3_combined_scale)

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a chess board position"""
        feat = DNNFeatures.board_to_features(board)
        return self.evaluate_full(feat)


def load_quantized_model(filepath: str):
    """
    Load a quantized model and return appropriate inference engine.

    Returns:
        QuantizedNNUEInference or QuantizedDNNInference
    """
    qw = QuantizedWeights.load(filepath)
    if qw.nn_type == "NNUE":
        return QuantizedNNUEInference(qw)
    else:
        return QuantizedDNNInference(qw)


def compare_quantized_vs_float(model: torch.nn.Module, nn_type: str,
                               num_positions: int = 100) -> Dict[str, float]:
    """
    Compare quantized model accuracy against float model.

    Args:
        model: Trained PyTorch model
        nn_type: "NNUE" or "DNN"
        num_positions: Number of random positions to test

    Returns:
        Dictionary with comparison metrics
    """
    from nn_inference import NNUEInference, DNNInference
    import random

    # Create both inference engines
    if nn_type == "NNUE":
        float_inference = NNUEInference(model)
    else:
        float_inference = DNNInference(model)

    qw = quantize_model(model, nn_type)
    if nn_type == "NNUE":
        quant_inference = QuantizedNNUEInference(qw)
    else:
        quant_inference = QuantizedDNNInference(qw)

    # Generate random positions
    errors = []
    max_error = 0

    for _ in range(num_positions):
        board = chess.Board()

        # Make some random moves
        num_moves = random.randint(5, 40)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)

        if board.is_game_over():
            continue

        # Evaluate with both engines
        float_score = float_inference.evaluate_board(board)
        quant_score = quant_inference.evaluate_board(board)

        error = abs(float_score - quant_score)
        errors.append(error)
        max_error = max(max_error, error)

    if not errors:
        return {'mean_error': 0, 'max_error': 0, 'positions_tested': 0}

    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': max_error,
        'positions_tested': len(errors),
        'mean_error_cp': np.mean(errors) * TANH_SCALE,  # Approximate centipawn error
        'max_error_cp': max_error * TANH_SCALE
    }


def print_quantization_stats(qw: QuantizedWeights):
    """Print statistics about quantized weights"""
    print("\n" + "=" * 60)
    print("QUANTIZATION STATISTICS")
    print("=" * 60)
    print(f"Network type: {qw.nn_type}")
    print(f"Input size: {qw.input_size}")
    print(f"Hidden size: {qw.hidden_size}")

    print("\nScale factors:")
    for name, scale in qw.scales.items():
        print(f"  {name}: {scale:.4f}")

    print("\nWeight ranges (quantized):")
    print(f"  ft_weight: [{qw.ft_weight.min()}, {qw.ft_weight.max()}] (int16)")
    print(f"  l1_weight: [{qw.l1_weight.min()}, {qw.l1_weight.max()}] (int8)")
    print(f"  l2_weight: [{qw.l2_weight.min()}, {qw.l2_weight.max()}] (int8)")
    print(f"  l3_weight: [{qw.l3_weight.min()}, {qw.l3_weight.max()}] (int8)")

    # Memory usage
    ft_bytes = qw.ft_weight.nbytes + qw.ft_bias.nbytes
    l1_bytes = qw.l1_weight.nbytes + qw.l1_bias.nbytes
    l2_bytes = qw.l2_weight.nbytes + qw.l2_bias.nbytes
    l3_bytes = qw.l3_weight.nbytes + qw.l3_bias.nbytes
    total_bytes = ft_bytes + l1_bytes + l2_bytes + l3_bytes

    print(f"\nMemory usage:")
    print(f"  Feature transform: {ft_bytes:,} bytes")
    print(f"  L1: {l1_bytes:,} bytes")
    print(f"  L2: {l2_bytes:,} bytes")
    print(f"  L3: {l3_bytes:,} bytes")
    print(f"  Total: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")
    print("=" * 60)


# Command-line interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Quantize NNUE/DNN model')
    parser.add_argument('model_path', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for quantized weights (default: model_quantized.bin)')
    parser.add_argument('--type', type=str, default='NNUE', choices=['NNUE', 'DNN'],
                        help='Network type')
    parser.add_argument('--compare', action='store_true',
                        help='Compare quantized vs float accuracy')
    parser.add_argument('--num-positions', type=int, default=1000,
                        help='Number of positions for comparison test')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    if args.type == "NNUE":
        model = NNUENetwork(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
    else:
        model = DNNNetwork(DNN_INPUT_SIZE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Quantize
    print(f"Quantizing {args.type} model...")
    qw = quantize_model(model, args.type)
    print_quantization_stats(qw)

    # Save
    output_path = args.output or args.model_path.replace('.pt', '_quantized.bin')
    qw.save(output_path)

    # Compare if requested
    if args.compare:
        print(f"\nComparing quantized vs float ({args.num_positions} positions)...")
        stats = compare_quantized_vs_float(model, args.type, args.num_positions)
        print(f"\nComparison results:")
        print(f"  Mean error: {stats['mean_error']:.6f} ({stats['mean_error_cp']:.2f} cp)")
        print(f"  Std error:  {stats['std_error']:.6f}")
        print(f"  Max error:  {stats['max_error']:.6f} ({stats['max_error_cp']:.2f} cp)")
        print(f"  Positions tested: {stats['positions_tested']}")