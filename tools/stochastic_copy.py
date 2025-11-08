import torch
import contextlib


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


# Define the custom autograd function
class StochasticCast(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, source: torch.Tensor, target_dtype: torch.dtype):
        # --- This forward pass contains the non-differentiable logic ---
        # It calculates the output value but doesn't need to track gradients itself.

        # Determine the intermediate FP32 tensor
        if source.dtype == torch.float64:
            # Clone to avoid modifying input if it requires grad and might be reused
            fp32_input = source.to(torch.float32, copy=True)
        elif source.dtype == torch.float32:
            fp32_input = source.clone()
        else:
            # If it's already the target dtype or another type (e.g., int),
            # just perform a standard cast. This path IS differentiable.
            return source.to(target_dtype)

        # Perform the stochastic rounding using bitwise ops
        # Use torch.no_grad() as gradients cannot be computed here anyway.
        with torch.no_grad():
            # Create random int32 (safer than using like on potentially modified fp32_input shape)
            random_int = torch.randint(0, (1 << 16), fp32_input.shape, dtype=torch.int32, device=fp32_input.device)

            # View input as int32 and add random integer
            result_int = fp32_input.view(dtype=torch.int32) + random_int

            # Mask off lower 16 bits (inplace bitwise_and_ is fine within no_grad)
            result_int.bitwise_and_(-65536)  # -65536 = FFFF0000 in signed int32

            # View back as float32
            result_fp32 = result_int.view(dtype=torch.float32)

        # Cast to the final target dtype
        final_result = result_fp32.to(target_dtype)
        # --- End of non-differentiable forward logic ---

        return final_result

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')  # Decorator for mixed precision compatibility
    def backward(ctx, grad_output: torch.Tensor):
        # --- This backward pass defines how gradients flow *through* the function ---
        # We treat the stochastic cast as having an identity gradient.
        # The gradient flowing back (`grad_output`) is passed through unmodified.
        # PyTorch handles casting the gradient back to the source tensor's dtype automatically.

        # Return gradient corresponding to 'source' input, and None for 'target_dtype'
        return grad_output, None


# Create a wrapper function to call the autograd Function easily
def to_stochastic(source: torch.Tensor, dtype=None):
    """
    Performs a stochastic cast to the target dtype using a custom autograd
    function to preserve gradient flow.
    """
    if dtype is None:
        raise ValueError("Target dtype must be specified for to_stochastic")

    if source is None or source.dtype == dtype:
        return source  # No operation needed, gradients preserved

    # Use the custom autograd function's apply method
    return StochasticCast.apply(source, dtype)
