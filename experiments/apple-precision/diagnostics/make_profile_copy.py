"""Create a timer-instrumented copy of the frozen r2 core for attribution only."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    build=json.loads((ROOT/'build-r2/build.json').read_text())
    assert not a.output.exists();(a.output/'native').mkdir(parents=True)
    for name,digest in build['sources'].items():
        assert sha(ROOT/name)==digest,name
        shutil.copy2(ROOT/name,a.output/name)
    path=a.output/'native/precision.cpp';s=path.read_text()
    s=s.replace('#include <vector>','#include <vector>\n#include <chrono>\n#include <sstream>\n#include <thread>')
    s=s.replace('thread_local char error_text[512]={0};', '''thread_local char error_text[512]={0};
struct ProfileEvent {const char* kind;int m,k,t,first,last;long long ns;size_t thread;};
std::mutex profile_mutex;std::vector<ProfileEvent> profile_events;
using Clock=std::chrono::steady_clock;
void profile_event(const char* kind,int m,int k,int t,int first,int last,Clock::time_point start){
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
    const auto thread=std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::lock_guard<std::mutex> lock(profile_mutex);profile_events.push_back({kind,m,k,t,first,last,ns,thread});
}''')
    anchor='''            x->x8.resize(n);x->scales.assign(time,0.f);x->sums.assign(time,0);'''
    assert s.count(anchor)==1;s=s.replace(anchor,'''            const auto quant_start=Clock::now();
'''+anchor)
    anchor='''            pack_dot_panels(*x);''';assert s.count(anchor)==1
    s=s.replace(anchor,'''            profile_event("quantize_and_allocate",p.m,p.k,time,0,p.m,quant_start);
            const auto pack_start=Clock::now();
            pack_dot_panels(*x);
            profile_event("pack",p.m,p.k,time,0,p.m,pack_start);''')
    anchor='''        else if(p.backend==2)apple_dot(p,x,y,first,last);''';assert s.count(anchor)==1
    s=s.replace(anchor,'''        else if(p.backend==2){const auto start=Clock::now();apple_dot(p,x,y,first,last);
            profile_event("dot_and_dequant_worker",p.m,p.k,x.t,first,last,start);}''')
    anchor='''int ip_abi(void){return 1;}''';assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'''
IP_EXPORT void ipa_profile_reset(){std::lock_guard<std::mutex> lock(profile_mutex);profile_events.clear();}
IP_EXPORT const char* ipa_profile_json(){
    static thread_local std::string result;std::ostringstream out;out<<"[";
    std::lock_guard<std::mutex> lock(profile_mutex);bool comma=false;
    for(const auto& e:profile_events){if(comma)out<<",";comma=true;
        out<<"{\\"kind\\":\\""<<e.kind<<"\\",\\"M\\":"<<e.m<<",\\"K\\":"<<e.k
           <<",\\"T\\":"<<e.t<<",\\"first\\":"<<e.first<<",\\"last\\":"<<e.last
           <<",\\"ns\\":"<<e.ns<<",\\"thread\\":"<<e.thread<<"}";
    }out<<"]";result=out.str();return result.c_str();
}
''')
    path.write_text(s)
    proof={'frozen_build_sha256':sha(ROOT/'build-r2/build.json'),'generator_sha256':sha(__file__),
           'scope':'Attribution only: preparation quantization/allocation, scalar pack and worker dot/dequant clocks; no arithmetic changes',
           'source_hashes':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    (a.output/'profile-copy.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__':main()
