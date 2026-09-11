"""Fresh combined B/startup recovery history; original measurements are read-only references."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import progressive_monitor as original
from progressive_continue_monitor import _validate_report

REVIEW_STEPS = (0,250,500,1000,1500,2000,2500)


class RecoveryMonitor(original.ProgressiveMonitor):
    def __init__(self,logdir,*,source_metadata=None,writer_factory=None):
        self.path = Path(logdir).resolve()
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError('Use a fresh combined TensorBoard directory')
        self.path.mkdir(parents=True,exist_ok=True)
        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter
            writer_factory = lambda path:SummaryWriter(str(path),flush_secs=10)
        self.factory,self.writers = writer_factory,{}
        self.cut_name,self.widths = 'Combined_B_startup_384x256',(384,256)
        self.total_updates,self.target = 2500,.99
        self.baseline,self.signature = None,None
        self.last_training_step,self.last_validation_step = 0,-1
        self.closed = False
        self.source_metadata = copy.deepcopy(source_metadata or {})
        self.details = self._writer('Details')
        self.details.add_custom_scalars({'Combined initialization recovery':{
            'Quality and progress':['Multiline',[r'^overview/Percent$']],
            'Matched original waveform error':['Multiline',['quality/mae','matched_original/quality/mae']],
            'Matched original mel error':['Multiline',['quality/mel','matched_original/quality/mel']],
            'Matched B waveform error':['Multiline',['quality/mae','matched_b/quality/mae']]}})
        self.details.add_text('Guide/Reading this recovery',
            'Fresh combined B/startup initialization, widths384/256, original frozen teacher. '
            'The first30,000 original fitting sources are consumed once in their original order. '
            'Each colored run is one metric from the combined candidate. Error reductions use combined step0; they are not fidelity. '
            'The99% cosine line is a milestone, not perceptual accuracy. Active/whistling RMS target100% '
            'of the teacher; higher and lower levels can both be wrong. Quiet cohorts overlap. '
            'All seven cohorts retain raw measurements in Details. Quality reviews are0,250,500,1000,1500,2000,2500. '
            'Raw losses update every step. Progress always uses2500 updates. Original measurements at matched '
            'steps are separate reference tags, never replayed training. Original250 is unavailable. '
            'B comparisons through2000 are separate matched_b tags. Adam is continuous throughout; '
            'this run stops at2500 for external review. Startup constraints are an initialization property, '
            'not an added training loss; later updates can change startup behavior.',0)
        self._json('monitor.json',{'version':'audiovae2_combined_recovery_monitor_v1','quality_steps':REVIEW_STEPS,
            'total_updates':2500,'baseline':'Authenticated combined step0','original_history_replayed':False,
            'automatic_next_cut':False})

    def log_training(self,record,step):
        if step != self.last_training_step+1 or record.get('step') != step:
            raise ValueError('Combined training history must contain each update once in order')
        super().log_training(record,step)
        for name in ('training_update_seconds','validation_seconds','waiting_seconds'):
            if name in record: self.details.add_scalar('performance/'+name,original._number(record[name],name,0),step)

    def log_validation(self,report,step,reference=None,b_reference=None):
        self._step(step)
        expected = REVIEW_STEPS[0] if self.last_validation_step < 0 else (
            REVIEW_STEPS[REVIEW_STEPS.index(self.last_validation_step)+1] if self.last_validation_step < 2500 else None)
        if step != expected or (step and step != self.last_training_step):
            raise ValueError('Combined review must follow the fixed milestones and completed training')
        signature = _validate_report(report,self.signature)
        if reference is not None: _validate_report(reference,signature)
        if b_reference is not None: _validate_report(b_reference,signature)
        aggregate = report['aggregate']
        if self.baseline is None:
            self.baseline,self.signature = dict(aggregate),signature
            self._json('combined-step0-baseline.json',{'aggregate':self.baseline,'panel':signature})
        cosine = aggregate.get('nonquiet_cosine_mean')
        self._overview('01 Active waveform cosine - milestone99%',None if cosine is None else 100*cosine,step)
        self._overview('02 Cosine milestone -99%',99.,step)
        omitted = []
        for key,label in original.ERRORS.items():
            if self.baseline[key] > 0: self._overview(label.replace('cut start','combined start'),100*(1-aggregate[key]/self.baseline[key]),step)
            else: omitted.append(key)
        for name,label in original.PASSES.items():
            row = report['quiet_regions']['regions'][name]
            if row['windows']: self._overview(label,100*(1-row['failed']/row['windows']),step)
        self._overview('12 Signal peak - limit100% full scale',100*aggregate['peak_abs_max'],step)
        if step == 0: self._overview('13 Training progress - target 100%',0.,step)
        rows = report.get('overview_window_metrics',{}).get('by_source',{})
        ratios = {}
        for name,label in (('active','14 Active RMS level - teacher matched100%'),('whistle','15 Whistling RMS level - teacher matched100%')):
            chosen = [r for sid,r in rows.items() if name == 'active' or
                'human_whistling_source_description' in self.source_metadata.get(sid,{}).get('verified_source_labels',[])]
            te = sum(r['active_teacher_energy'] for r in chosen); pe = sum(r['active_student_energy'] for r in chosen)
            ratio = math.sqrt(pe/te) if te > 0 else None
            ratios[name+'_rms_ratio'] = ratio
            self._overview(label,None if ratio is None else 100*ratio,step)
            if ratio is not None: self.details.add_scalar('quality/'+name+'_rms_ratio',ratio,step)
        for prefix,current in (('',report),('matched_original/',reference),('matched_b/',b_reference)):
            if current is None: continue
            for key,value in current['aggregate'].items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    self.details.add_scalar(prefix+'quality/'+key,value,step)
            for name,row in current['quiet_regions']['regions'].items():
                for key,value in row.items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool):
                        self.details.add_scalar(prefix+'quiet/'+name+'/'+key,value,step)
                for key,value in row['failure_categories'].items():
                    self.details.add_scalar(prefix+'quiet/'+name+'/'+key,value,step)
        self._json('latest-validation.json',{'step':step,'aggregate':aggregate,'quiet_regions':report['quiet_regions'],
            'amplitude_ratios':ratios,'original_reference_available':reference is not None,
            'b_reference_available':b_reference is not None,
            'omitted_zero_baseline_reductions':omitted})
        self.last_validation_step = step
        self.flush()
