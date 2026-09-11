"""Read-only, teacher-anchored quiet residual decomposition for saved students."""
from __future__ import annotations
import torch
from audiovae_student.fusion_evaluation import _preserved_evaluation


def phase_decomposition(prediction, target, *, quiet_rms=0.001, min_cycles=8):
    """Use complete 40 ms cycles so DC/480/1920 terms are orthogonal.

    The caller aligns the start to the actual 1920-sample latent grid and
    supplies valid audio only. Spectral periodicity does not imply a tone or
    a causal defect. The unstructured remainder includes genuine noise.
    """
    p, t = prediction.reshape(-1).double(), target.reshape(-1).double()
    if p.shape != t.shape or not bool(torch.isfinite(p).all() and torch.isfinite(t).all()):
        raise ValueError('Expected identical finite valid waveform shapes')
    cycles = p.numel() // 1920
    p, t = p[:cycles*1920].reshape(-1,1920), t[:cycles*1920].reshape(-1,1920)
    quiet = t.square().mean(1).sqrt() <= quiet_rms
    p, t = p[quiet], t[quiet]
    result = {'quiet_cycles_40ms':int(quiet.sum()),'minimum_cycles':min_cycles,
              'teacher_quiet_rms_threshold':quiet_rms,'discarded_tail_samples':prediction.numel()%1920}
    if p.shape[0] < min_cycles:
        return result
    e = p-t
    dc = e.mean()
    template = e.mean(0)
    phase480 = e.reshape(-1,4,480).mean((0,1))
    powers = {'dc':dc.square(), 'phase480_ac':(phase480-dc).square().mean(),
        'phase1920_additional':(template-phase480.repeat(4)).square().mean(),
        'remaining':(e-template).square().mean()}
    total = e.square().mean()
    indices = torch.tensor([480,960,1440],device=p.device)
    adjacent = e[:,1:]-e[:,:-1]
    ordinary = torch.ones(1919,dtype=torch.bool,device=p.device)
    ordinary[indices-1]=False
    result.update(residual_rms=float(total.sqrt()),student_rms=float(p.square().mean().sqrt()),
        teacher_rms=float(t.square().mean().sqrt()),residual_dc=float(dc),
        power_components={k:float(v) for k,v in powers.items()},
        power_fractions={k:float(v/total.clamp_min(1e-30)) for k,v in powers.items()},
        decomposition_absolute_error=float(abs(total-sum(powers.values()))),
        interior_block_boundary_residual_jump_rms=float((e[:,indices]-e[:,indices-1]).square().mean().sqrt()),
        interior_block_boundary_teacher_jump_rms=float((t[:,indices]-t[:,indices-1]).square().mean().sqrt()),
        ordinary_neighbor_residual_jump_rms=float(adjacent[:,ordinary].square().mean().sqrt()))
    return result


def diagnose_components(engine,crops,metadata):
    """Same input and target per arm; capture the existing head's actual input."""
    rows=[]
    with _preserved_evaluation(engine.model):
        for crop in crops:
            info=metadata.get(crop.source_id,{})
            recorded=[]
            hook=engine.model.output.register_forward_pre_hook(
                lambda module,args:recorded.append(args[0].detach()))
            try:
                z=crop.latents.to(engine.device)
                prediction=engine.model(z)
            finally:
                hook.remove()
            expected=z.shape[-1]*1920
            if prediction.shape != (1,1,expected):
                raise ValueError('Decoder sample count changed')
            # Interior historical panels have29latentcontext; keep alignment
            # while excluding the first40ms cycle from this separate analysis.
            skip=1920 if crop.context_start_frame>0 else 0
            start=crop.scored_slice.start+skip;stop=crop.scored_slice.stop
            if stop<=start:continue
            p=prediction[...,start:stop]
            t=crop.teacher_audio[...,start:stop].to(engine.device)
            row={'source_id':crop.source_id,'start_frame':crop.start_frame,'metadata':info,
                'diagnostic_context_exclusion_samples':skip,'valid_samples':p.numel(),
                'raw_mae':float((p-t).abs().mean()),'student_peak':float(p.abs().max()),
                'teacher_peak':float(t.abs().max()),'latent_rms':float(z.square().mean().sqrt()),
                'quiet':phase_decomposition(p,t)}
            if recorded:
                h=recorded[0][...,start//480:stop//480]
                row['head_input_rms']=float(h.square().mean().sqrt()) if h.numel() else None
                row['head_input_phase_rms']=[float(h[...,j::4].square().mean().sqrt())
                    if h[...,j::4].numel() else None for j in range(4)]
            rows.append(row)
    return {'format_version':1,'step':engine.step,'rows':rows,
        'interpretation':'Descriptive phase decomposition, not proof of an architectural cause',
        'alignment':'40ms latent grid, complete teacher-quiet cycles, no artificial padded tails',
        'parameter_updates':0}
