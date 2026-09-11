"""Independently stage the requested Cantonese addition with identical rules."""
from pathlib import Path
import acquire_fleurs_heldout_languages as acquisition
from acquire_heldout_languages import ROOT,write

subset=ROOT/'cantonese-addition'
subset.mkdir(exist_ok=True)
for name in ['indic-report.json','indic-manifest.jsonl']:
    write(subset/name,(ROOT/name).read_bytes())
acquisition.ROOT=subset
acquisition.LANGS={'yue_hant_hk':'yue'}
acquisition.main()
