---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: audio
    dtype:
      audio:
        decode: false
  - name: title
    dtype: string
  - name: description
    dtype: string
  - name: tags
    list: string
  - name: username
    dtype: string
  - name: freesound_id
    dtype: uint64
  - name: license
    dtype:
      class_label:
        names:
          '0': CC0-1.0
          '1': CC-BY-4.0
          '2': CC-BY-3.0
          '3': CC-BY-NC-3.0
          '4': CC-BY-NC-4.0
          '5': CC-Sampling+
  - name: attribution_required
    dtype:
      class_label:
        names:
          '0': 'No'
          '1': 'Yes'
  - name: commercial_use
    dtype:
      class_label:
        names:
          '0': 'No'
          '1': 'Yes'
  splits:
  - name: train
    num_bytes: 83499892601
    num_examples: 50000
  download_size: 83502496395
  dataset_size: 83499892601
---
