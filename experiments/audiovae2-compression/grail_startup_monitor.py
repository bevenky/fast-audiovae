"""Fresh G-plus-startup2k charts using the existing fifteen quality curves."""
import copy
from pathlib import Path
import startup_anchor_monitor as old


class RecoveryMonitor(old.RecoveryMonitor):
    def __init__(self,logdir,*,source_metadata=None,writer_factory=None):
        self.path=Path(logdir).resolve()
        if self.path.exists() and any(self.path.iterdir()):raise FileExistsError('Use a fresh recovery event directory')
        self.path.mkdir(parents=True,exist_ok=True)
        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter
            writer_factory=lambda path:SummaryWriter(str(path),flush_secs=10)
        self.factory,self.writers=writer_factory,{}
        self.cut_name,self.widths='G_plus_startup_corrected_384x256',(384,256)
        self.total_updates,self.target=2000,.99
        self.baseline,self.signature=None,None
        self.last_training_step,self.last_validation_step=0,-1
        self.closed=False;self.source_metadata=copy.deepcopy(source_metadata or {})
        self.details=self._writer('Details')
        self.details.add_custom_scalars({'G plus startup recovery':{
            'Quality and progress':['Multiline',[r'^overview/Percent$']],
            'Actual motion and normal correction':['Multiline',[
                'quiet_update/accepted_displacement_norm','quiet_update/normal_accepted_norm']]}})
        self.details.add_text('Guide/Reading this recovery',
            'Fresh original teacher factory plus sealed G and constrained native upsampler initialization. '
            'Widths384/256; all90 group tensors train, original teacher and outer stages stay frozen. '
            'The original24,000 ordinary sources are consumed once. Six calibration starts recur separately. '
            'All12 startup inequalities use the qualified complete projection and bounded normal correction. '
            'Adam moments advance once per ordinary batch. Base fraction1 with a normal flag is a repaired '
            'projected step, not an unchanged Adam step; actual motion and normal norms are separate. '
            'Linear q_correction_norm and q_projection_cosine exclude subsequent normal correction. '
            'No development example selects an update. Six-anchor retention does not guarantee unseen startup. '
            'The15 colored curves use the actual fresh candidate step0; relative error reductions are not fidelity. '
            'The99% cosine line is a milestone. Active/whistling RMS targets100% of teacher. Seven quiet cohorts overlap. '
            'Reviews occur0/250/500/1000/1500/2000, with raw losses every update and progress always out of2000. '
            'No pilot state is resumed; Adam remains continuous through1000; stop at2000 for review. '
            'No later width cut or promotion starts automatically.',0)
        self._json('monitor.json',{'version':'audiovae2_grail_startup_monitor_v1','quality_steps':old.REVIEW_STEPS,
            'total_updates':2000,'baseline':'Authenticated fresh G-plus-startup initializer',
            'pilot_history_replayed':False,'automatic_next_cut':False})
