# Activation-Aware Weight Quantization for OPT-1.3B

This project provides a Python implementation for simulating the quantization of neural network weights in PyTorch models, particularly focusing on transformer models like those from the Hugging Face Transformers library. The primary goal is to reduce the model size and potentially increase the inference speed while trying to maintain the accuracy of the model.


## Usage

1. **Quantize All Weights**: This will quantize all the weights of the model using a specified bit width and group size.

2. **Quantize Salient Weights**: This will quantize only the top 1% salient weights based on the provided feature importance, keeping them in FP16 format.

3. **Random Weight Quantization**: This randomly selects 1% of the weights and quantizes them, keeping the selected weights in FP16 format.

4. **Scale-Up Quantization**: Applies a scale factor to the salient weights before quantization and scales them down post-quantization.

5. **Auto-Scale Quantization**: Automatically finds the optimal scale for quantization to minimize the output deviation caused by quantization.

## Example

To run the quantization simulation, execute the script `quantize.py`:

This will perform the quantization process and print out the model's perplexity and size before and after quantization.

