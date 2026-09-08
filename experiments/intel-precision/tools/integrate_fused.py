"""Create isolated mixed-precision copies of the accepted fused CPU operators."""
from pathlib import Path
import argparse,hashlib,json
BUNDLE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
ROOT=args.output_dir.resolve();REPO=args.repo.resolve()
if ROOT.exists():raise ValueError('Use a fresh generated-source directory')
expected=json.loads((BUNDLE/'pins/integration.json').read_text())
for name,digest in expected['source_sha256'].items():
 if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:raise ValueError('Upstream fused source hash mismatch: '+name)
ROOT.mkdir(parents=True)

def change(s,old,new):
 assert s.count(old)==1,(old[:100],s.count(old))
 return s.replace(old,new)

def common(s,old_domain,new_domain):
 s=change(s,'#include "stage_dw.h"','#include "stage_dw.h"\n#include "precision.h"')
 s=change(s,old_domain,new_domain)
 return s

stage_path=REPO/'experiments/cpu-stage/stage/custom_op.cpp'
s=stage_path.read_text()
s=common(s,'"fast.audiovae.stage.experimental"','"fast.audiovae.precision.stage.experimental"')
s=change(s,'  std::array<Unit,3> units;','  std::array<Unit,3> units;\n  int precision_mode;\n  std::array<std::shared_ptr<void>,3> precision;')
s=change(s,'    plans.resize(q+1);','''    precision_mode=integer("precision_mode");
    require(precision_mode==8||precision_mode==16,"Precision mode must be 8 or 16");
    for(int u=0;u<3;++u){
      precision[u]=std::shared_ptr<void>(ip_create(c,c,units[u].pw.data(),precision_mode,1),ip_destroy_plan);
      require(bool(precision[u]),ip_last_error());
    }
    plans.resize(q+1);''')
s=change(s,'''          const void* p=work.plans[n].get();
          status=fx_run_range(p,z.pw.data(),pre,z.pb.data(),a,b,0,fx_blocks(p));
          require(status==0,"Complete-K pointwise failed");''','''          std::unique_ptr<void,decltype(&ip_destroy_input)> prepared(
              ip_prepare(k.precision[u].get(),pre,n),ip_destroy_input);
          require(bool(prepared),ip_last_error());
          require(ip_run_rows(k.precision[u].get(),prepared.get(),b,0,k.c)==0,ip_last_error());
          for(int oc=0;oc<k.c;++oc)for(int it=0;it<n;++it){
            const size_t offset=static_cast<size_t>(oc)*n+it;
            const float dot_bias=b[offset]+z.pb[oc];
            b[offset]=a[offset]+dot_bias;
          }''')
s=change(s,'// Experimental three-residual-unit pipeline. CPU FP32 only; no runtime default.', '// Experimental mixed-precision matrices in the accepted fused stage. FP32 IO and nonlinearities.')
(ROOT/'stage_precision.cpp').write_text(s)

up_path=REPO/'experiments/cpu-stage/upsample/custom_op.cpp'
s=up_path.read_text()
s=common(s,'"fast.audiovae.upsample.experimental"','"fast.audiovae.precision.upsample.experimental"')
s=change(s,'    std::array<Unit,3> units;','''    std::array<Unit,3> units;
    int precision_mode;
    std::shared_ptr<void> current_precision,previous_precision;
    std::array<std::shared_ptr<void>,3> precision;''')
s=change(s,'        residual_plans.resize(q+1);projection_plans.resize(q/Stride+1);','''        precision_mode=integer("precision_mode");
        require(precision_mode==8||precision_mode==16,"Precision mode must be 8 or 16");
        current_precision=std::shared_ptr<void>(ip_create(ProjectionChannels,ProjectionChannels,current_w.data(),precision_mode,1),ip_destroy_plan);
        require(bool(current_precision),ip_last_error());
        previous_precision=std::shared_ptr<void>(ip_create(ProjectionChannels,ProjectionChannels,previous_w.data(),precision_mode,1),ip_destroy_plan);
        require(bool(previous_precision),ip_last_error());
        for(int u=0;u<3;++u){
            precision[u]=std::shared_ptr<void>(ip_create(Channels,Channels,units[u].pw.data(),precision_mode,1),ip_destroy_plan);
            require(bool(precision[u]),ip_last_error());
        }
        residual_plans.resize(q+1);projection_plans.resize(q/Stride+1);''')
s=change(s,'''                require(up_projection_run(work.projection[1].get(),k.previous_w.data(),gathered,seeded)==0,
                        "Real previous-column projection failed");''','''                std::unique_ptr<void,decltype(&ip_destroy_input)> prepared(
                    ip_prepare(k.previous_precision.get(),gathered,1),ip_destroy_input);
                require(bool(prepared),ip_last_error());
                require(ip_run_rows(k.previous_precision.get(),prepared.get(),seeded,0,ProjectionChannels)==0,ip_last_error());''')
s=change(s,'''                const void* projection=work.projection[low_n].get();
                require(up_projection_run(projection,k.current_w.data(),post,b)==0,"Current raw projection failed");
                require(up_projection_run(projection,k.previous_w.data(),post,pre)==0,"Previous raw projection failed");''','''                {
                    std::unique_ptr<void,decltype(&ip_destroy_input)> prepared(
                        ip_prepare(k.current_precision.get(),post,low_n),ip_destroy_input);
                    require(bool(prepared),ip_last_error());
                    require(ip_run_rows(k.current_precision.get(),prepared.get(),b,0,ProjectionChannels)==0,ip_last_error());
                    require(ip_run_rows(k.previous_precision.get(),prepared.get(),pre,0,ProjectionChannels)==0,ip_last_error());
                }''')
s=change(s,'''                    const void* plan=work.residual[n].get();
                    require(fx_run_range(plan,z.pw.data(),pre,z.pb.data(),a,b,0,fx_blocks(plan))==0,
                            "Ordered residual pointwise failed");''','''                    std::unique_ptr<void,decltype(&ip_destroy_input)> prepared(
                        ip_prepare(k.precision[u].get(),pre,n),ip_destroy_input);
                    require(bool(prepared),ip_last_error());
                    require(ip_run_rows(k.precision[u].get(),prepared.get(),b,0,Channels)==0,ip_last_error());
                    for(int oc=0;oc<Channels;++oc)for(int it=0;it<n;++it){
                        const size_t offset=static_cast<size_t>(oc)*n+it;
                        const float dot_bias=b[offset]+z.pb[oc];
                        b[offset]=a[offset]+dot_bias;
                    }''')
s=change(s,'// Isolated stage4 upsampling and three-unit residual pipeline, CPU FP32 only.', '// Experimental mixed-precision matrices in the accepted fused upsampling stage. FP32 IO.')
(ROOT/'upsample_precision.cpp').write_text(s)
meta={'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (stage_path,up_path)},'generated_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'stage_precision.cpp',ROOT/'upsample_precision.cpp')},'preserved':'Causal halos, phase carry, segmentation, FP32 Snake/depthwise/bias/residual, output interface. Matrix products alone changed. All nonlinearities retain the accepted implementations.'}
(ROOT/'integration.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta))

if meta['generated_sha256']!=expected['generated_sha256']:raise ValueError('Generated operator bytes differ from frozen integration')
