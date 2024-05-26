import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
import gc
from activation_calib import *
from utils import *

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    # assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w

@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_group_size,
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

@torch.no_grad()
def pseudo_quantize_model_salient_weight_fp16(model, w_bit, q_group_size, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            # Find 1% of the salient weight channels according to importance
            num_channels = m.weight.data.size(1)
            k = max(1, num_channels // 100)
            _, outlier_indices = torch.topk(importance, k)
            assert outlier_indices.dim() == 1

            # Back up the values of the salient weight channels
            outlier = m.weight.data[:, outlier_indices].clone()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            # Restore the 1% salient weight channels to their original FP16 values
            m.weight.data[:, outlier_indices] = outlier


@torch.no_grad()
def pseudo_quantize_model_random_weight_fp16(model, w_bit, q_group_size, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            # Randomly choose 1% of the weight channels
            num_channels = m.weight.data.size(1)
            k = max(1, num_channels // 100)
            outlier_mask = torch.randint(0, num_channels, (k,))
            assert outlier_mask.dim() == 1

            # Back up the values of the selected weight channels
            outlier = m.weight.data[:, outlier_mask].clone()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            # Restore the 1% selected weight channels to their original FP16 values
            m.weight.data[:, outlier_mask] = outlier


@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(
    model, w_bit, q_group_size, input_feat, scale_factor
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            # Find 1% of the salient weight channels
            num_channels = m.weight.data.size(1)
            k = max(1, num_channels // 100)
            _, outlier_mask = torch.topk(importance, k)
            assert outlier_mask.dim() == 1

            # To simulate applying the scale factor, we can simply multiply it before quantization, and then divide by the scale factor after quantization.
            # Scale up the values of the salient weight channels
            m.weight.data[:, outlier_mask] *= scale_factor
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            # Scale back down the values of the salient weight channels
            m.weight.data[:, outlier_mask] /= scale_factor

@torch.no_grad()
def auto_scale_block(module, name, w_bit, q_group_size, input_feat):
    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)

        # Initialize the best_error, best_ratio and best_scales
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid

            # Calculate the scales by the formula: scales = s_x^ratio
            scales = s_x.pow(ratio)
            assert scales.shape == s_x.shape

            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:
                scales = scales.to(fc.weight.device)

                # Scale up the values of the weight channels
                fc.weight.mul_(scales)

                fc.weight.data = pseudo_quantize_tensor(
                    fc.weight.data, w_bit, q_group_size
                )

                # Scale back down the values of the weight channels
                fc.weight.div_(scales)

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # attention input
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # attn out
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(
        module.self_attn.out_proj, [module.self_attn.out_proj], inp
    )
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # fc1
    inp = input_feat[name + ".fc1"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # fc2
    inp = input_feat[name + ".fc2"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    scale_fc_fc(module.fc1, module.fc2, final_scales)

@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(model, w_bit, q_group_size, input_feat):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

if __name__ == "__main__":
    # Original FP16 model
    model_path = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    # Evaluate the model
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=32, group_size=128)
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")

    # Quantize all weights
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)

    # Evaluate the model
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    input_feat = get_calib_feat(model, tokenizer)

    # Quantize top 1% activations
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    pseudo_quantize_model_salient_weight_fp16(model, w_bit=3, q_group_size=128, input_feat=input_feat)

    # Evaluate the model
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")

    # Randomly quantize 1% of weights
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    pseudo_quantize_model_random_weight_fp16(model, w_bit=3, q_group_size=128, input_feat=input_feat)

    # Evaluate the model
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")

    # Quantize using scaleup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    pseudo_quantize_model_weight_scaleup(model, w_bit=3, q_group_size=128, input_feat=input_feat, scale_factor=2)

    # Evaluate the model
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")

    # Find the optimal scale within predefined search space
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    pseudo_quantize_model_weight_auto_scale(model, w_bit=3, q_group_size=128, input_feat=input_feat)

    # Evaluate the model
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")