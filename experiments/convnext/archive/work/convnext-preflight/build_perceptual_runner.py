from pathlib import Path
root=Path('work/fast-audiovae/experiments/convnext/audiovae_student')
s=(root/'balance_comparison.py').read_text()
s=s.replace('BalanceComparisonConfig','PerceptualComparisonConfig').replace('run_balance_comparison','run_perceptual_comparison')
s=s.replace('total_steps: int = 500','total_steps: int = 1000').replace('expected_parent_step: int = 1009','expected_parent_step: int = 250')
s=s.replace('    candidate_kind: str = "mel_cap"\n    quiet_phase_share: float = .02\n','')
s=s.replace('            if key in ("candidate_kind", "quiet_phase_share"):\n                continue\n','')
a=s.index('        if self.total_steps > 500:')
b=s.index('\n\ndef start_fingerprints',a)
s=s[:a]+'''        if self.total_steps > 1000 or self.evaluation_interval >= self.total_steps or self.evaluation_interval > 250:
            raise ValueError("Perceptual trial requires at most1000 updates and review by250")

    @property
    def arms(self):
        return ("control", "perceptual")


def new_arm(parent_state, config, device, *, capped, readiness_inputs, crops):
    saved=deepcopy(parent_state)
    if saved.get('quiet_phase') is not None or saved.get('perceptual_trial') is not None:
        raise ValueError('Perceptual trial requires an explicitly selected plain reconstruction parent')
    model=StudentDecoder(StudentConfig(**saved['model_config'])).to(device)
    cls=PerceptualTrialEngine if capped else DistillationEngine
    kwargs={'trial_config': PerceptualTrialConfig(max_updates=config.total_steps,review_update=config.evaluation_interval)} if capped else {}
    engine=cls(model,config=DistillationTrainingConfig(**saved['config']),
        loss_config=DistillationLossConfig(**saved['loss_config']),
        balancer_config=GradientBalancerConfig(**saved['balancer']['config']),
        discriminators=AudioDiscriminators(DiscriminatorConfig(**saved['discriminator_config'])),**kwargs)
    engine.load_state_dict(saved)
    if capped:
        provenance=engine.enable_trial(readiness_inputs['report'],readiness_inputs['policy'],crops=crops,
            current=readiness_inputs['current'],previous=readiness_inputs['previous'],evidence=readiness_inputs['evidence'])
    else:
        fork=replace(engine.config,total_steps=config.total_steps,warmup_steps=0,freeze_normalization_step=0,
            learning_rate_schedule='constant_after_warmup',warmup_start_learning_rate=None,
            final_learning_rate=engine.config.learning_rate)
        provenance=engine.fork_reconstruction(fork)
    return engine,provenance
''' +s[b:]
s=s.replace('(\"model\", \"optimizer\", \"discriminators\", \"discriminator_optimizer\", \"crop_rng\")','(\"model\", \"optimizer\", \"discriminators\", \"crop_rng\")')
a=s.index('def compare_evaluations(');b=s.index('\ndef _implementation',a)
s=s[:a]+'''def compare_evaluations(reports, previous=None):
    control,candidate=(reports[a]['groups']['all'] for a in ('control','perceptual'))
    checks={}
    for key in ('teacher_waveform_mean','teacher_mel_mean','quiet_residual_rms_mean','quiet_residual_rms_p95'):
        checks[key+'_within_10_percent_of_control']=candidate[key] is not None and control[key] is not None and candidate[key]<=1.10*control[key]
    checks['no_added_clipping']=candidate['student_clipped_samples']<=control['student_clipped_samples']
    checks['no_added_active_collapse']=candidate['active_nearzero_count']<=control['active_nearzero_count']
    checks['level_error_increase_at_most_1db']=candidate['nonquiet_absolute_level_error_db_median']<=control['nonquiet_absolute_level_error_db_median']+1
    conditions={}
    for group,c in reports['control']['groups'].items():
        if group.startswith('condition/'):
            v=reports['perceptual']['groups'][group]
            ratio=v['teacher_waveform_mean']/max(c['teacher_waveform_mean'],1e-12)
            delta=(v['nonquiet_cosine_mean']-c['nonquiet_cosine_mean']) if v['nonquiet_cosine_mean'] is not None and c['nonquiet_cosine_mean'] is not None else None
            conditions[group]={'waveform_error_ratio':ratio,'cosine_delta':delta}
    checks['no_condition_collapse']=all(v['waveform_error_ratio']<=1.10 and (v['cosine_delta'] is None or v['cosine_delta']>=-.05) for v in conditions.values())
    return {'checks':checks,'safety_checks_passed':all(checks.values()),'conditions':conditions,
        'mel_improvement_fraction':1-candidate['teacher_mel_mean']/control['teacher_mel_mean'],
        'automatic_promotion':False,'interpretation':'Controlled perceptual experiment. Passing rollback checks is not quality acceptance.'}

''' +s[b:]
s=s.replace('def _implementation(candidate_kind="mel_cap"):', 'def _implementation():')
s=s.replace('names = ("balance_comparison",', 'names = ("perceptual_comparison", "perceptual_trial_engine", "perceptual_readiness", "balance_comparison",')
s=s.replace('    if candidate_kind == "quiet_phase":\n        names += ("quiet_phase", "quiet_phase_engine")\n','')
a=s.index('\ndef component_mel_share');b=s.index('\n\n@_exclusive_run',a)
s=s[:a]+s[b:]
s=s.replace('run_prefix="balance-v1", heldout_corpus=None,\n        phase_policy=None', 'run_prefix="perceptual-v1", heldout_corpus=None,\n        readiness_inputs=None, review_before_continuation=False')
s=s.replace('    windows, training_rows = tuple(windows), tuple(training_rows)', '''    if not isinstance(readiness_inputs,dict):
        raise ValueError('Exact parent readiness artifacts are required')
    if readiness_inputs['policy']['parent_checkpoint_sha256'] != parent_checkpoint_sha256:
        raise ValueError('Readiness policy refers to another parent')
    windows, training_rows = tuple(windows), tuple(training_rows)
    required=parent['engine']['config']['adversarial_samples']
    if any(w.valid_output_samples48k<required for w in windows):
        raise ValueError('Every planned scored crop must support the adversarial window')''')
s=s.replace('"kind": "paired_unique_balance_comparison"','"kind": "paired_unique_perceptual_comparison"')
s=s.replace('_implementation(config.candidate_kind)','_implementation()')
a=s.index('        "runtime": runtime_identity(device),');b=s.index('\n    directory = Path(output_dir)',a)
s=s[:a]+'''        "runtime": runtime_identity(device), "readiness_artifacts":readiness_inputs,
        "comparison":{"control":"unchanged reconstruction", "perceptual":"500-step ramp into MPD/MRD adversarial and feature matching"},
        "rollback_policy":{"waveform_mel_quiet_ratio_max":1.10,"level_error_increase_db_max":1.,
            "condition_waveform_ratio_max":1.10,"condition_cosine_delta_min":-.05,"automatic_promotion":False}}))''' +s[b:]
s=s.replace('capped=name == candidate_name, phase_policy=phase_policy','capped=name == candidate_name, readiness_inputs=readiness_inputs, crops=heldout_crops')
s=s.replace('"Paired model, optimizer or EMA starting values differ"','"Paired student, generator optimizer, discriminator weights or EMA starting values differ"')
s=s.replace('        _restore_rng(saved["rng"])','''        _restore_rng(saved["rng"])
        if review_before_continuation:
            if step != config.evaluation_interval or not evaluation['comparison']['safety_checks_passed']:
                raise ValueError('Matched-control safety checks do not permit continuation')
            evidence=evaluation_evidence(heldout_crops,evaluation['arms']['perceptual']['rows'],
                run_identity=identity,model_config=engines['perceptual'].model.config.to_dict())
            review=engines['perceptual'].review_trial(evidence,crops=heldout_crops)
            _atomic_json(directory/'safety-review.json',review)
            if not review['continuation_authorized']:
                raise ValueError('Perceptual trial failed its baseline safety review')''')
s=s.replace('    stop_step = min(config.total_steps, step + max_updates) if max_updates else config.total_steps','''    stop_step = min(config.total_steps, step + max_updates) if max_updates else config.total_steps
    if engines['perceptual'].trial_review is None:
        stop_step=min(stop_step,config.evaluation_interval)''')
a=s.index('                    if name == "mel-cap"');b=s.index('                _restore_rng(rng_after)',a)
s=s[:a]+s[b:]
s=s.replace('"perceptual_training_started": False', '"perceptual_training_started": True')
s=s.replace('"new_updates_per_arm": step - first_step', '"new_updates_per_arm": step - first_step')
s=s.replace('"paused_at_explicit_bound"','"paused_for_safety_review"')
s=s.replace('from .gradient_balancer import GradientBalancerConfig, MelGradientCapConfig','''from .gradient_balancer import GradientBalancerConfig
from .perceptual_trial_engine import PerceptualTrialEngine, PerceptualTrialConfig
from .perceptual_readiness import evaluation_evidence
from .balance_comparison import component_mel_share''')
# Include all component norms when reporting the perceptual stage, where GAN/FM are active.
s=s.replace('metrics[name]["actual_teacher_mel_share"] = component_mel_share(metrics[name])', '''norms=[metrics[name].get(k+'/scaled_norm',0.) for k in ('teacher_waveform','teacher_mel','feature_matching','adversarial')]
                    metrics[name]['actual_teacher_mel_share']=norms[1]/sum(norms) if sum(norms) else 0.''')
s=s.replace('"""Paired finite-data continuations with one shared frozen-teacher batch.\n\nThe comparison changes only an explicit output-gradient balancing policy.','"""Paired finite-data perceptual trial with an explicit review boundary.\n\nThe comparison changes only the declared training objective.')
(root/'perceptual_comparison.py').write_text(s)
