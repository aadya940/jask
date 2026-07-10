from abc import ABC, abstractmethod

import jax


class Op(ABC):
    """Base Op."""

    pass


class BlockParallelOp(Op):
    """Ops which can easily be divided into Page/Frame based
    subproblems.

    Only needs an implementation of subproblems. Automatically,
    parallelised.
    """

    @abstractmethod
    def forward_block(self, *input_blocks: jax.Array) -> jax.Array:
        """Complete In-Memory Operation.

        Examples
        --------
        def forward_block(self, block_1, block_2):
            return block_1 @ block_2
        """
        pass

    @abstractmethod
    def index_map(self, out_idx: tuple) -> list[tuple]:
        """Given an Output Block, which Input coordinates does it need and
        how many calls to foward pass.

        Examples
        --------
        def index_map(out_idx):
            i, j = out_idx
            return [((i, k), (k, j)) for k in range(self.k_blocks)]
        """
        pass

    @abstractmethod
    def combine(self, acc: jax.Array, partial: jax.Array) -> jax.Array:
        """Incremental (pairwise) reduction of one new partial into the
        running accumulator for an output block. Called once per entry in
        index_map's result, not on a collected list, this keeps only the
        accumulator and one partial resident at a time, instead of holding
        every partial for an output block simultaneously.

        Examples
        --------
        def combine(self, acc, partial):
            return acc + partial
        """
        pass

    @abstractmethod
    def backward_block(self, d_out_block, *input_blocks) -> tuple[jax.Array]:
        """Defines VJP gradient for JAX reverse mode autodiff compatibility.

        Examples
        --------
        def backward_block(self, d_out_block, block_1, block_2):
            dA = d_out_block @ block_2.T
            dB = block_1.T @ d_out_block
            return (dA, dB)
        """
        pass

    @abstractmethod
    def output_shape(self, *input_shapes) -> tuple:
        """Defines Output Shape.

        Examples
        --------
        def output_shape(self, block_1, block_2):
            return (block_1.shape[0], block_2.shape[1])
        """
        pass

    @classmethod
    def from_inputs(cls, *inputs):
        """Build an op instance given the actual input BlockedArrays.
        Default: no-arg construction. Override when the op needs shape-derived
        params, e.g. Dot needs k_blocks."""
        return cls()


class CustomOp(Op):
    """For ops that cannot be block-parallelised (e.g. fft, softmax, sort).

    Instead of block-level methods, the user implements complete
    forward/backward passes that get the OOCAlgorithm passed in as a
    toolkit for block I/O, memory tracking, etc.
    """

    @abstractmethod
    def forward(self, algo, *inputs):
        """Run the full forward pass. Return a BlockedArray of the output."""
        pass

    @abstractmethod
    def backward(self, algo, inputs, d_out):
        """Run the full backward pass. Return a tuple of input gradients."""
        pass

    @abstractmethod
    def output_shape(self, *input_shapes) -> tuple:
        """Declare the output's shape given input shapes."""
        pass

    @classmethod
    def from_inputs(cls, *inputs):
        """Build an op instance given the actual input BlockedArrays.
        Default: no-arg. Override if the op needs shape-derived config."""
        return cls()
