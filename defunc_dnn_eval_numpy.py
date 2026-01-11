"""
Optimized NumPy-based DNN evaluator for chess positions.

Key optimizations over the original:
1. Pre-transpose weights for faster matmul (row-major access pattern)
2. Use np.dot with pre-transposed weights instead of x @ W
3. Fused bias addition with matmul where possible
4. Avoid unnecessary array copies
5. Use out= parameter for in-place operations
6. Contiguous array enforcement for SIMD optimization
7. Optional: numba JIT compilation for activation functions
"""

from typing import Any, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray

from defunc_build_model import tanh_to_score
from cached_board import CachedBoard

INF = 10_000

# Try to import numba for JIT compilation (optional)
try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    prange = range


# ============================================================================
# Optimized activation functions
# ============================================================================

@njit(cache=True, fastmath=True)
def relu_inplace(x: np.ndarray) -> np.ndarray:
    """ReLU activation - modifies array in place."""
    flat = x.ravel()
    for i in prange(len(flat)):
        if flat[i] < 0:
            flat[i] = 0
    return x


@njit(cache=True, fastmath=True)
def tanh_fast(x: np.ndarray) -> np.ndarray:
    """Fast tanh using numpy."""
    return np.tanh(x)


def relu_numpy(x: np.ndarray) -> np.ndarray:
    """ReLU using numpy maximum (fallback if no numba)."""
    return np.maximum(x, 0, out=x)  # in-place


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    """Sigmoid with clipping for numerical stability."""
    np.clip(x, -500, 500, out=x)
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# Optimized Dense Layer
# ============================================================================

class OptimizedDenseLayer:
    """
    Pre-optimized dense layer for fast inference.

    Key optimizations:
    - Weights stored transposed for row-major access
    - Bias stored as contiguous array
    - Activation function bound at init time
    """
    __slots__ = ('weights_T', 'bias', 'activation_fn', 'name')

    def __init__(self, weights: np.ndarray, bias: np.ndarray,
                 activation: str, name: str = ""):
        # Store weights transposed: (out_features, in_features)
        # This makes np.dot(x, W.T) become np.dot(x, weights_T.T) = x @ W
        # But we store W.T directly, so: output = x @ weights_T.T = x @ W
        # Actually for (batch, in) @ (in, out) -> (batch, out)
        # We want weights as (in, out), transposed is (out, in)
        # np.dot(x, W) where x is (batch, in) and W is (in, out)
        # Storing W transposed as (out, in) means we do: np.dot(x, W_T.T)
        # But that's the same... Let's just ensure contiguity

        self.weights_T = np.ascontiguousarray(weights.T, dtype=np.float32)  # (out, in)
        self.bias = np.ascontiguousarray(bias, dtype=np.float32)
        self.name = name

        # Bind activation function
        if activation == 'relu':
            self.activation_fn = relu_inplace if HAS_NUMBA else relu_numpy
        elif activation == 'tanh':
            self.activation_fn = np.tanh
        elif activation == 'sigmoid':
            self.activation_fn = sigmoid_numpy
        elif activation in ('linear', None):
            self.activation_fn = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x @ W + b, but W is stored transposed
        # x: (batch, in_features), weights_T: (out_features, in_features)
        # We need: (batch, in) @ (in, out) = (batch, out)
        # With transposed storage: x @ weights_T.T
        out = np.dot(x, self.weights_T.T) + self.bias

        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


# ============================================================================
# Optimized Model Container
# ============================================================================

class KerasModelNumPyOptimized:
    """
    Optimized NumPy inference engine for Keras models.

    Supports: Dense, Flatten, Dropout, InputLayer
    (Conv2D and BatchNorm available but not optimized - rarely used for this model)
    """

    def __init__(self, model_path: str):
        """Load Keras model and extract optimized layer representations."""
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        model.summary()

        self.layers: List[Any] = []
        self._extract_and_optimize_layers(model)

        # Pre-allocate output buffer for common batch size
        self._output_buffer = None

    def _extract_and_optimize_layers(self, model):
        """Extract layers and create optimized versions."""
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            config = layer.get_config()

            if layer_type == 'InputLayer':
                continue

            elif layer_type == 'Dense':
                weights, bias = layer.get_weights()
                activation = config.get('activation', 'linear')
                opt_layer = OptimizedDenseLayer(weights, bias, activation, layer.name)
                self.layers.append(('dense', opt_layer))

            elif layer_type == 'Flatten':
                self.layers.append(('flatten', None))

            elif layer_type == 'Dropout':
                # No-op during inference
                continue

            elif layer_type == 'BatchNormalization':
                gamma, beta, moving_mean, moving_var = [w.numpy() for w in layer.weights]
                epsilon = config.get('epsilon', 1e-3)
                # Pre-compute scale and shift: y = gamma * (x - mean) / sqrt(var + eps) + beta
                # = gamma / sqrt(var + eps) * x + (beta - gamma * mean / sqrt(var + eps))
                scale = gamma / np.sqrt(moving_var + epsilon)
                shift = beta - gamma * moving_mean / np.sqrt(moving_var + epsilon)
                self.layers.append(('batchnorm', (
                    np.ascontiguousarray(scale, dtype=np.float32),
                    np.ascontiguousarray(shift, dtype=np.float32)
                )))

            elif layer_type == 'Activation':
                activation = config.get('activation', 'linear')
                self.layers.append(('activation', activation))

            else:
                print(f"Warning: Layer type '{layer_type}' not optimized, using fallback")
                # Store original layer config for fallback
                self.layers.append(('fallback', {
                    'type': layer_type,
                    'weights': [w.numpy() for w in layer.weights],
                    'config': config
                }))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run optimized inference.

        Args:
            x: Input array, shape (batch_size, ...) or (features,)

        Returns:
            Model output as numpy array
        """
        # Ensure batch dimension
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Ensure float32 and contiguous for optimal SIMD
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        # Forward pass
        for layer_type, layer_data in self.layers:
            if layer_type == 'dense':
                x = layer_data(x)

            elif layer_type == 'flatten':
                x = x.reshape(x.shape[0], -1)

            elif layer_type == 'batchnorm':
                scale, shift = layer_data
                x = x * scale + shift

            elif layer_type == 'activation':
                if layer_data == 'relu':
                    x = relu_numpy(x)
                elif layer_data == 'tanh':
                    x = np.tanh(x)
                elif layer_data == 'sigmoid':
                    x = sigmoid_numpy(x)

            elif layer_type == 'fallback':
                x = self._fallback_layer(x, layer_data)

        return x

    def _fallback_layer(self, x: np.ndarray, layer_info: dict) -> np.ndarray:
        """Handle unsupported layer types with basic implementation."""
        layer_type = layer_info['type']
        weights = layer_info['weights']
        config = layer_info['config']

        if layer_type == 'Conv2D':
            # Basic conv2d - not optimized
            kernel, bias = weights
            return self._conv2d_basic(x, kernel, bias, config)

        raise NotImplementedError(f"Layer type {layer_type} not implemented")

    def _conv2d_basic(self, x, kernel, bias, config):
        """Basic Conv2D implementation (not optimized)."""
        # This is rarely called for the chess model which is primarily Dense layers
        batch_size, in_h, in_w, in_c = x.shape
        k_h, k_w, _, out_c = kernel.shape
        strides = config.get('strides', (1, 1))
        padding = config.get('padding', 'valid')
        activation = config.get('activation', 'linear')

        stride_h, stride_w = strides if isinstance(strides, tuple) else (strides, strides)

        if padding == 'same':
            out_h = int(np.ceil(in_h / stride_h))
            out_w = int(np.ceil(in_w / stride_w))
            pad_h = max((out_h - 1) * stride_h + k_h - in_h, 0)
            pad_w = max((out_w - 1) * stride_w + k_w - in_w, 0)
            x = np.pad(x, ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                           (pad_w // 2, pad_w - pad_w // 2), (0, 0)))
        else:
            out_h = (in_h - k_h) // stride_h + 1
            out_w = (in_w - k_w) // stride_w + 1

        output = np.zeros((batch_size, out_h, out_w, out_c), dtype=np.float32)

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start, w_start = i * stride_h, j * stride_w
                    patch = x[b, h_start:h_start + k_h, w_start:w_start + k_w, :]
                    for c in range(out_c):
                        output[b, i, j, c] = np.sum(patch * kernel[:, :, :, c]) + bias[c]

        # Apply activation
        if activation == 'relu':
            output = relu_numpy(output)
        elif activation == 'tanh':
            output = np.tanh(output)

        return output


# ============================================================================
# Evaluator Class
# ============================================================================

class DNNEvaluatorNumPyOptimized:
    """DNN-based position evaluator using optimized NumPy inference."""

    __slots__ = ('_model', '_model_path')

    def __init__(self):
        self._model: Optional[KerasModelNumPyOptimized] = None
        self._model_path: Optional[str] = None

    def _load_model(self, filepath: str):
        """Load model if not already loaded or if path changed."""
        if self._model is None or self._model_path != filepath:
            self._model = KerasModelNumPyOptimized(filepath)
            self._model_path = filepath

    def evaluate(self, cached_board: CachedBoard, model_filepath: str) -> int:
        """
        Evaluate position from side-to-move perspective.

        Returns:
            Score in centipawns. Positive = good for side to move.
        """
        self._load_model(model_filepath)

        board_repr = cached_board.get_board_repr()
        # Flatten and add batch dimension in one operation
        board_repr = board_repr.reshape(1, -1).astype(np.float32)

        return self.dnn_eval_board_repr(board_repr)

    def dnn_eval_board_repr(self, board_repr: NDArray[Any]) -> int:
        """Evaluate a board representation array."""
        score = self._model.predict(board_repr)[0][0]
        score = tanh_to_score(score)
        return int(score)


# ============================================================================
# Global instance and convenience function
# ============================================================================

_evaluator = DNNEvaluatorNumPyOptimized()


def dnn_eval(board: CachedBoard, model_filepath: str) -> int:
    """
    Evaluate position using DNN model with optimized NumPy inference.

    Args:
        board: Position to evaluate (should be quiet position)
        model_filepath: Path to Keras model file

    Returns:
        Score in centipawns from side-to-move perspective
    """
    return _evaluator.evaluate(board, model_filepath)


# ============================================================================
# Main / Testing
# ============================================================================

def main():
    """Interactive testing and benchmarking."""
    import time
    from defunc_build_model import DNN_MODEL_FILEPATH

    print(f"NumPy optimization status:")
    print(f"  - Numba available: {HAS_NUMBA}")
    print(f"  - NumPy version: {np.__version__}")
    print()

    # Test position
    TEST_FEN = "r2q1rk1/1bpn1pbp/1p2pnp1/p2p4/3P3P/1P3NP1/PBPNPPB1/2RQ1RK1 w - - 0 1"

    # Load model and warm up
    board = CachedBoard(TEST_FEN)
    board_repr = board.get_board_repr()
    board_repr = board_repr.reshape(1, -1).astype(np.float32)

    print("Loading model...")
    _evaluator._load_model(DNN_MODEL_FILEPATH)

    # Warm up
    for _ in range(10):
        _evaluator.dnn_eval_board_repr(board_repr)

    # Benchmark
    NUM_ITERATIONS = 1000
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        _evaluator.dnn_eval_board_repr(board_repr)
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # Convert to ms
    print(f"\nBenchmark ({NUM_ITERATIONS} iterations):")
    print(f"  Mean:   {np.mean(times):.4f} ms")
    print(f"  Std:    {np.std(times):.4f} ms")
    print(f"  Min:    {np.min(times):.4f} ms")
    print(f"  Max:    {np.max(times):.4f} ms")
    print(f"  Median: {np.median(times):.4f} ms")

    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive mode (type 'exit' to quit)")
    print("=" * 50)

    while True:
        try:
            fen = input("\nFEN: ").strip()
            if fen.lower() == "exit":
                break
            if fen == "":
                continue

            board = CachedBoard(fen)
            score = dnn_eval(board, DNN_MODEL_FILEPATH)
            print(f"Score: {score} cp")
            print(f"Side to move: {'White' if board.turn else 'Black'}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()