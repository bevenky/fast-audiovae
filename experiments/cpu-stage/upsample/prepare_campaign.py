"""Prepare frozen Intel decoder screen/final configs. No model execution."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ACCEPTED_SOURCE = 'e188d0609795d256627b4e39b632d5c5ca064256d410899ecb05ac4eb6301bc2'
SCREEN_UIDS = ['hi_in_00099_1919', 'en_us_00103_1779', 'pt_br_00672_1745']
FINAL_UIDS = ['hi_in_00099_1919', 'bn_in_00633_1820', 'gu_in_00926_1827',
              'ta_in_00494_1946', 'te_in_00114_1911', 'kn_in_00584_1904',
              'en_us_00103_1779', 'es_419_00646_1781', 'fr_fr_00129_1717',
              'pt_br_00672_1745']


def load_harness(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('frozen_decoder_campaign_harness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(*, source_config, source_config_sha256, harness, graph, graph_sha256,
            library, library_sha256, output_dir):
    source_config, harness = Path(source_config).resolve(), Path(harness).resolve()
    graph, library, output = Path(graph).resolve(), Path(library).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError('Output directory must not exist; preserve previous campaign evidence')
    if any(path.is_relative_to(output) for path in (source_config,harness,graph,library)):
        raise ValueError('Source artifacts must be outside the new output directory')
    h = load_harness(harness)
    config, verified = h.read_config(source_config, source_config_sha256)
    h.require_hash(graph, graph_sha256)
    h.require_hash(library, library_sha256)
    if config['expected_case_count'] != 60 or len(config['expected_uids']) != 60:
        raise ValueError('Preserve the original complete 60-case corpus')
    if config.get('timing_uids') != FINAL_UIDS or not set(FINAL_UIDS) <= set(config['expected_uids']):
        raise ValueError('Original ten timing UIDs and their order are required')
    byname = {v['name']:v for v in config['models']}
    if not {'audio_stock','stage_mkl','mimi'} <= set(byname):
        raise ValueError('Original audio_stock, stage_mkl and mimi definitions are required')
    if 'upsample_stage4' in byname:
        raise ValueError('Source config is already an upsample campaign')
    parent = source_config.parent
    resolve = lambda value: h.relative_path(parent,value)
    metadata_counts={}
    for kind in ('audio','mimi'):
        metadata=resolve(config[kind+'_cases']).with_suffix('.json')
        timed_ids=json.loads(metadata.read_text()).get('timed_ids')
        if timed_ids != config['expected_uids']:
            raise ValueError('The original sixty metadata clips and their order are required for ' + kind)
        metadata_counts[kind]=len(timed_ids)
    old_graph = resolve(byname['stage_mkl']['path'])
    h.require_hash(old_graph,ACCEPTED_SOURCE)
    old_libraries = byname['stage_mkl'].get('custom_libraries')
    if (byname['stage_mkl'].get('custom_library') or not isinstance(old_libraries,list)
            or len(old_libraries) != 3 or len({resolve(v) for v in old_libraries}) != 3):
        raise ValueError('Accepted stage_mkl requires its original three distinct libraries')
    if graph == old_graph or library in {resolve(v) for v in old_libraries}:
        raise ValueError('The derivative graph and extra library must be new artifacts')
    if byname['stage_mkl']['kind'] != 'audio' or not byname['stage_mkl']['causal']:
        raise ValueError('Accepted causal AudioVAE2 contract required')
    artifact_hashes = {}

    def register(path, digest):
        path=Path(path).resolve()
        if path.is_relative_to(output): raise ValueError('An input artifact is inside the new output directory')
        name=Path(os.path.relpath(path,output)).as_posix()
        if name in artifact_hashes and artifact_hashes[name] != digest.lower():
            raise ValueError('Conflicting hashes after path normalization')
        artifact_hashes[name]=digest.lower()
        return name

    # Preserve even source hashes no longer referenced by the four selected
    # models. The public harness will recheck every one of them on each run.
    for name,digest in verified.items(): register(resolve(name),digest)
    def rebase(value):
        path=resolve(value)
        return register(path,verified[value])

    base=copy.deepcopy(config)
    for kind in base['kind_contracts']:
        key=kind+'_cases'
        if key in base: base[key]=rebase(base[key])
    if base.get('custom_library'): base['custom_library']=rebase(base['custom_library'])
    models=[]
    for name in ('audio_stock','stage_mkl','mimi'):
        row=copy.deepcopy(byname[name]);row['path']=rebase(row['path'])
        if row.get('custom_library'): row['custom_library']=rebase(row['custom_library'])
        for key in ('custom_libraries','external_data'):
            if key in row: row[key]=[rebase(v) for v in row[key]]
        models.append(row)
    selected=copy.deepcopy(models[1])
    selected.update(name='upsample_stage4',path=register(graph,graph_sha256))
    selected['custom_libraries']=[*models[1]['custom_libraries'],register(library,library_sha256)]
    selected['external_data']=[register(path,h.sha(path)) for path in sorted(h.model_external_data(graph))]
    models.insert(2,selected)
    base['models']=models
    base['artifact_sha256']=artifact_hashes
    # Keep old provenance with an explicit parent label. It describes the
    # accepted source campaign, not the newly selected graph or new library.
    if 'candidate_manifest_sha256' in base:
        base['source_candidate_manifest_sha256']=base.pop('candidate_manifest_sha256')
    base['upsample_campaign_provenance']={
        'source_config':Path(os.path.relpath(source_config,output)).as_posix(),
        'source_config_sha256':source_config_sha256.lower(),
        'source_stage_mkl_sha256':ACCEPTED_SOURCE,
        'selected_graph_sha256':graph_sha256.lower(),
        'selected_library_sha256':library_sha256.lower(),
        'preparation_script_sha256':h.sha(__file__),
        'harness_sha256':h.sha(harness),
        'original_corpus_and_artifact_hashes_preserved':True,
        'metadata_timed_ids_match_all_expected_uids':True,
        'metadata_timed_id_counts':metadata_counts,
        'model_execution':False,'timing_execution':False,'gpu_used':False}
    configs={}
    for name,uids in (('screen',SCREEN_UIDS),('final',FINAL_UIDS)):
        value=copy.deepcopy(base);value['timing_uids']=list(uids)
        value['scope']=(
            'Original complete 60-case corpus retained. Screen timing uses the unchanged Hindi, English and Portuguese clips. '
            'For these frozen companion files, --validation timed also validates all sixty metadata clips. '
            'The current campaign skips this redundant screen and runs the final configuration only.'
            if name=='screen' else
            'Original complete 60-case corpus retained. Run --validation all for full-corpus numerical acceptance; '
            'timing uses the unchanged original ten representative clips. CPU decoder only.')
        configs[name]=value
    output.mkdir(parents=True,exist_ok=False)
    hashes={}
    try:
        for name,value in configs.items():
            path=output/(name+'.json')
            with path.open('x') as f: f.write(json.dumps(value,indent=2)+'\n')
            digest=h.sha(path)
            checked,after=h.read_config(path,digest)
            if checked['expected_uids'] != config['expected_uids'] or checked['expected_case_count'] != 60:
                raise RuntimeError('Generated config changed corpus coverage')
            if after != value['artifact_sha256']: raise RuntimeError('Artifact verification changed')
            hashes[path.name]=digest
        h.require_hash(source_config,source_config_sha256)
        h.require_hash(graph,graph_sha256);h.require_hash(library,library_sha256)
        report={
            'status':'complete','configs':hashes,'model_order':[v['name'] for v in models],
            'expected_case_count':60,'screen_timing_uids':SCREEN_UIDS,'final_timing_uids':FINAL_UIDS,
            'screen_validation':'--validation timed checks all sixty metadata clips for this corpus',
            'final_validation':'--validation all checks all sixty clips',
            'metadata_timed_id_counts':metadata_counts,
            'current_execution_plan':'Skip redundant screen; final validation of sixty clips and timing of ten original clips',
            'source_artifact_count':len(verified),'generated_artifact_count':len(artifact_hashes),
            'provenance':base['upsample_campaign_provenance'],
            'inference_executed':False,'default_promoted':False}
        path=output/'preparation.json'
        with path.open('x') as f: f.write(json.dumps(report,indent=2)+'\n')
        with (output/'SHA256SUMS').open('x') as f:
            for name,digest in hashes.items(): f.write(digest+'  '+name+'\n')
            f.write(h.sha(path)+'  preparation.json\n')
    except BaseException:
        # Preserve any newly created evidence rather than overwriting or
        # silently retrying after a failed integrity check.
        raise
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-config','source-config-sha256','harness','graph','graph-sha256',
                 'library','library-sha256','output-dir'):
        p.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(p.parse_args())),indent=2))
