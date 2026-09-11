"""Reproduce AudioVAE2 compression budgets without importing a neural runtime.

Reads frozen Python syntax and saved static arithmetic, not model parameters.
No model, dataset, inference, training, benchmark or remote operation is used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_defaults(path):
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AudioVAEConfig")
    return {n.target.id: ast.literal_eval(n.value) for n in cls.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def history_bounds(strides, dilations_by_stage, initial_history=6, final_history=6):
    """Exact extremal input-frame dependency from known convolution support.

    Causal transpose k=2s yields earliest input floor(n/s)-1; residual stack
    contributes six times summed dilations, and initial/final convs six taps.
    This is support arithmetic, not a model run or perceptual delay estimate.
    """
    product = 1
    for s in strides:
        product *= s
    earliest = []
    for phase in range(product):
        position = phase - final_history
        for stride, dilations in reversed(list(zip(strides, dilations_by_stage))):
            position = (position - 6*sum(dilations)) // stride - 1
        earliest.append(position-initial_history)
    return {"earliest_latent_offset_min": min(earliest), "earliest_latent_offset_max": max(earliest),
            "maximum_past_latent_frames": -min(earliest), "samples_per_latent": product,
            "future_latent_frames": 0,
            "initial_history_frames": initial_history, "final_history_frames": final_history,
            "note": "Offsets are in this box's input frames. Values vary over output polyphase positions; this describes receptive support, not added streaming latency."}


def audit(root):
    source = root/"work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py"
    cost_path = root/"outputs/intel-iteration2/data/architecture-costs.json"
    cfg = config_defaults(source)
    saved = json.loads(cost_path.read_text())["audio_vae2"]
    assert cfg["depthwise"] and not cfg["use_noise_block"]
    assert cfg["latent_dim"] == 64 and cfg["decoder_dim"] == 2048
    assert cfg["decoder_rates"] == [8, 6, 5, 2, 2, 2]
    initial = 25*64*7 + 25*64*2048
    final = 48000*32*7
    stages = []
    rate = 25
    for i, stride in enumerate(cfg["decoder_rates"]):
        cin, cout = cfg["decoder_dim"]//2**i, cfg["decoder_dim"]//2**(i+1)
        out_rate = rate*stride
        item = {"stage": i+1, "module": f"decoder.model.{i+2}",
            "input_channels": cin, "input_rate_hz": rate,
            "output_channels": cout, "output_rate_hz": out_rate,
            "sample_rate_condition_module": f"decoder.sr_cond_model.{i+2}",
            "contract": "Stage prehook receives post-conditioning input; whole-stage output is after all residual units.",
            "transpose": {"module": f"decoder.model.{i+2}.block.1", "kernel": 2*stride,
                "stride": stride, "left_input_state_frames": 1, "right_trim": stride,
                "output_length_multiplier": stride, "future_input_frames": 0},
            "upsample_mac_per_second": rate*cin*cout*2*stride,
            "residual_pointwise_mac_per_second": 3*out_rate*cout*cout,
            "residual_depthwise_mac_per_second": 3*out_rate*cout*7,
            "upsample_snake_values_per_second": rate*cin,
            "residual_snake_values_per_second": 6*out_rate*cout,
            "residual_state_float_values": 78*cout,
            "units": [{"dilation": dilation, "module": f"decoder.model.{i+2}.block.{u+2}",
                "left_history_frames": 6*dilation, "input_output_channels": cout,
                "rate_hz": out_rate, "pointwise_mac_per_second": out_rate*cout*cout,
                "depthwise_mac_per_second": out_rate*cout*7,
                "snake_values_per_second": 2*out_rate*cout}
                for u, dilation in enumerate([1, 3, 9])]}
        item["total_mac_per_second"] = sum(item[k] for k in ["upsample_mac_per_second",
            "residual_pointwise_mac_per_second", "residual_depthwise_mac_per_second"])
        assert item["upsample_mac_per_second"] == saved["stages"][i]["upsample_mac_per_second"]
        assert item["residual_pointwise_mac_per_second"] == saved["stages"][i]["residual_pointwise_mac_per_second"]
        assert item["residual_depthwise_mac_per_second"] == saved["stages"][i]["depthwise_mac_per_second"]
        stages.append(item); rate = out_rate
    total = initial+final+sum(s["total_mac_per_second"] for s in stages)
    snake = 48000*32+sum(s["upsample_snake_values_per_second"]+s["residual_snake_values_per_second"] for s in stages)
    assert total == round(saved["GMAC_per_audio_second"]*1e9)
    groups = {name: sum(s[key] for s in stages) for name, key in [
        ("upsampling", "upsample_mac_per_second"), ("residual_pointwise", "residual_pointwise_mac_per_second"),
        ("residual_depthwise", "residual_depthwise_mac_per_second")]}
    groups["initial_and_final_convolutions"] = initial+final
    candidates = []
    def add(name, family, modified_stages, saving, snake_saving, history, details):
        candidates.append({"name": name, "family": family, "modified_stages": modified_stages,
            "mac_per_second": total-saving, "gmac_per_second": (total-saving)/1e9,
            "mac_saved_per_second": saving, "mac_reduction_percent": 100*saving/total,
            "snake_values_saved_per_second": snake_saving, "snake_reduction_percent": 100*snake_saving/snake,
            "latent_history": history, "details": details,
            "measured_cpu_gain": None, "quality_validated": False})
    normal_history = history_bounds(cfg["decoder_rates"], [[1, 3, 9]]*6)
    for selected in [[3, 4, 5, 6], [1, 2, 3], [2, 3, 4], [1, 2, 3, 4, 5, 6]]:
        chosen = [s for s in stages if s["stage"] in selected]
        saving = sum((s["residual_pointwise_mac_per_second"]+s["residual_depthwise_mac_per_second"])//3 for s in chosen)
        sine = sum(s["residual_snake_values_per_second"]//3 for s in chosen)
        history = history_bounds(cfg["decoder_rates"], [[1, 9] if i+1 in selected else [1, 3, 9] for i in range(6)])
        add("remove_d3_unit_stages_"+"_".join(map(str, selected)), "A_unit_deletion", selected,
            saving, sine, history, "Initialize from d1 and d9 units, then jointly adapt BOTH to match the entire teacher three-unit stack output. No one-to-one retained-layer matching. Stage shapes/rates/zero boundaries preserved; initial function and receptive support changed.")
    chosen = stages
    saving = sum((s["residual_pointwise_mac_per_second"]+s["residual_depthwise_mac_per_second"])*2//3 for s in chosen)
    sine = sum(s["residual_snake_values_per_second"]*2//3 for s in chosen)
    add("three_to_one_residual_group_all_stages", "A_group_depth_distillation", list(range(1, 7)),
        saving, sine, history_bounds(cfg["decoder_rates"], [[9]]*6),
        "One residual unit jointly learns each complete teacher three-unit stack. d9 initialization is a specified example, not a proven optimum. Every stage interface/rate preserved; less temporal support. Whole-decoder MAC reduction29.15%, not30%+. Finalconv/tanh untouched.")
    for selected in [[3, 4], [3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]:
        chosen = [s for s in stages if s["stage"] in selected]
        saving = sum((s["residual_pointwise_mac_per_second"]+s["residual_depthwise_mac_per_second"])//2 for s in chosen)
        sine = sum(s["residual_snake_values_per_second"]//2 for s in chosen)
        add("half_residual_branch_channels_stages_"+"_".join(map(str, selected)), "B_internal_channel_pruning", selected,
            saving, sine, normal_history, "All three residual units retained. Select r=C/2 channels within each residual branch, use r-channel Snake/DW/Snake then C-by-r pointwise output and the full-C identity skip. No new dense entry projection. Preserve stage I/O width, rates, conditioning, transpose, all dilations and final conv/tanh. Reconstruct sliced effective weights before weight-normalization reparameterization.")
    saving = sum(s["residual_pointwise_mac_per_second"]//2 for s in stages)
    add("quarter_rank_residual_pointwise_all_stages", "B_internal_linear_factorization", list(range(1, 7)),
        saving, 0, normal_history, "Replace each C-by-C pointwise by r-by-C followed by C-by-r with r=C/4, no intervening activation. Cost2Cr is halfC²; r=C/2 would save no matrix MACs. Truncated SVD initializes an approximation, not exact inherited function. Must retain two factors at runtime; multiplying them into a dense exported weight forfeits the reduction.")
    # Larger teacher stage2-through4 box. Internal boundaries may change,
    # outer 1024@200Hz ->128@12000Hz stays identical, as do all other stages.
    group_reference_mac = sum(s["total_mac_per_second"] for s in stages[1:4])
    group_reference_snake = sum(s["upsample_snake_values_per_second"]+s["residual_snake_values_per_second"] for s in stages[1:4])
    group_candidates = []
    for units, dilations in [(3, [1, 3, 9]), (2, [1, 9]), (1, [9])]:
        internal = []; group_mac = 0; group_snake = 0
        for stage_index, cin, cout in [(2, 1024, 256), (3, 256, 128), (4, 128, 128)]:
            old = stages[stage_index-1]; fout=old["output_rate_hz"]; fin=old["input_rate_hz"]; stride=old["transpose"]["stride"]
            up = fin*cin*cout*2*stride; residual = units*fout*(cout*cout+7*cout)
            sine = fin*cin+2*units*fout*cout
            group_mac += up+residual; group_snake += sine
            internal.append({"stage": stage_index, "input_channels": cin, "output_channels": cout,
                "input_rate_hz": fin, "output_rate_hz": fout, "residual_units": units,
                "initial_dilations": dilations, "upsample_mac_per_second": up,
                "residual_mac_per_second": residual, "total_mac_per_second": up+residual,
                "snake_values_per_second": sine})
        history = history_bounds(cfg["decoder_rates"], [[1, 3, 9] if i not in [2, 3, 4] else dilations for i in range(1, 7)])
        name = f"stage2_to4_group_internal_halfwidth_{units}_units"
        add(name, "C_group_internal_width_and_depth", [2, 3, 4], group_reference_mac-group_mac,
            group_reference_snake-group_snake, history,
            "Match the COMPLETE teacher stage2-through4 black box output, not corresponding internal layers. Preserve outer1024@200Hz ->128@12000Hz, latent encoder, stages1/5/6 and finalconv/tanh. Internal stage2/3 widths256/128 replace512/256; stage4 output128 unchanged. Inherit selected effective weights and jointly adapt the entire group. Rate conditioning inside the group is selected consistently; outer conditioning can remain frozen.")
        candidates[-1]["internal_stages"] = internal
        candidates[-1]["group_reference_mac_per_second"] = group_reference_mac
        candidates[-1]["group_candidate_mac_per_second"] = group_mac
        candidates[-1]["group_mac_reduction_percent"] = 100*(group_reference_mac-group_mac)/group_reference_mac
        candidates[-1]["teacher_residual_unit_count"] = 9
        candidates[-1]["student_residual_unit_count"] = 3*units
        candidates[-1]["group_teacher_history"] = history_bounds([6, 5, 2], [[1, 3, 9]]*3, 0, 0)
        candidates[-1]["group_student_history"] = history_bounds([6, 5, 2], [dilations]*3, 0, 0)
        group_candidates.append(name)
    # Hypothetical full-width reduction, deliberately outside preserved-stage contract.
    slim_initial = 25*64*7+25*64*1024
    slim_final = 48000*16*7
    slim_stages = sum(s["upsample_mac_per_second"]//4+s["residual_pointwise_mac_per_second"]//4+
                      s["residual_depthwise_mac_per_second"]//2 for s in stages)
    add("half_stage_widths_including_stem_and_final_input", "C_stage_interface_change", list(range(1, 7)),
        total-(slim_initial+slim_final+slim_stages), snake//2, normal_history,
        "Reference budget only, not first pilot. All decoder widths halve; stage interfaces and final-convolution input width change. Restoring each teacher interface with projections would incur additional uncounted cost and require a different design.")
    return {"version": "audiovae2_static_compression_budget_v1", "model_operations_executed": 0,
        "scope": "Canonical neural contractions and Snake element evaluations, batch one per audio second. Static computation only; no claim of RTF or quality.",
        "inputs": {str(source.relative_to(root)): sha(source), str(cost_path.relative_to(root)): sha(cost_path)},
        "config": cfg, "decoder_module_inventory": {"initial_depthwise": "decoder.model.0", "initial_pointwise": "decoder.model.1",
            "stages": [s["module"] for s in stages], "final_snake": "decoder.model.8", "final_waveform_convolution": "decoder.model.9", "final_tanh": "decoder.model.10"},
        "baseline": {"mac_per_audio_second": total, "gmac_per_audio_second": total/1e9,
            "snake_values_per_audio_second": snake, "mac_groups": groups,
            "mac_group_percentages": {k: 100*v/total for k, v in groups.items()},
            "initial_convolution_mac_per_second": initial, "final_convolution_mac_per_second": final,
            "latent_history": normal_history}, "stages": stages, "candidates": candidates,
        "excluded": ["Bias/residual/rate-conditioning arithmetic", "Snake arithmetic cost beyond element count", "Tanh and normalization",
            "Memory movement, channel selection, matrix packing, dispatch, allocation", "Streaming kernel implementation and repeated chunk warmup"],
        "first_material_group_candidate": "stage2_to4_group_internal_halfwidth_3_units",
        "group_boundary": {"input": "decoder.model.2 output (stage1 output), before unchanged stage2 sample-rate conditioning", "input_channels": 1024, "input_rate_hz": 200,
            "output": "decoder.model.5 output (after all stage4 residual units)", "output_channels": 128, "output_rate_hz": 12000, "length_multiplier": 60,
            "unchanged_outside": ["encoder", "decoder.model.0", "decoder.model.1", "decoder.model.2", "decoder.model.6", "decoder.model.7", "decoder.model.8", "decoder.model.9", "decoder.model.10"],
            "training_target": "T_group(x) versus S_group(x) with exact shared x and aligned whole-group output; differentiate through frozen downstream decoder for waveform supervision. No projection needed at external boundaries."},
        "selection_basis": "Primary group distillation narrows only internal stage2–4 interfaces, retaining all9 residualunits, teacher dilations and receptive support:42.45%wholedecoder canonical MAC opportunity. The later9-to6-unit candidate saves another7.57%relative to this width-only candidate while reducing temporal support. If initialized from an accepted width-only student later, the ORIGINAL full teacher remains the distillation target. No quality or runtime promise; stage-hotspot evidence still governs pilot choice."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    result["analysis_script_sha256"] = sha(Path(__file__))
    output = args.out or root/"outputs/audiovae2-compression-plan/static-budget.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(output), "baseline_gmac": result["baseline"]["gmac_per_audio_second"],
        "baseline_snake_elements": result["baseline"]["snake_values_per_audio_second"],
        "candidates": [{k: c[k] for k in ["name", "gmac_per_second", "mac_reduction_percent", "snake_reduction_percent"]} for c in result["candidates"]]}, indent=2))
