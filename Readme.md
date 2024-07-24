# CLAPScore for language-queried audio source separation

This is the implementation of the paper [A Reference-free Metric for Language-Queried Audio Source Separation using Contrastive Language-Audio Pretraining](https://arxiv.org/abs/2407.04936), which proposes the CLAPScore metric for the language-queried audio source separation.

## Format of Evaluated Data

The evaluated data includes the `text_queries.csv` and the `audio_dir`.

The `text_queries.csv` records the text queries used for separation, which should be in the following format:
```csv
audio_file, text_query
sample1.wav, text query of sample1.wav
sample2.wav, text query of sample2.wav
...
```

The `audio_dir` is the path to the audio files, which should be in the following format:
```
audio_dir:
    sample1.wav
    sample2.wav
    ...
```

## Pretrained Checkpoint of CLAP

We employ the pretrained checkpoint of CLAP to calculate the CLAPScore metric, which is available at [music_speech_audioset_epoch_15_esc_89.98.pt](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint)

## How to Use

The evaluation process is in the `main.py`. Please replace the `text_queries` and `audio_dir` into yours in `main.py`. Then, you can run the `main.py` to obtain the evaluation results.

## Citation

```
@article{xiao2024CLAPScore,
  title={A Reference-free Metric for Language-Queried Audio Source Separation using Contrastive Language-Audio Pretraining},
  author={Xiao, Feiyang and Guan, Jian and Zhu, Qiaoxi and Liu, Xubo and Wang, Wenbo and Qi, Shuhan and Zhang, Kejia and Sun, Jianyuan and Wang, Wenwu},
  journal={arXiv preprint arXiv:2407.04936},
  year={2024}
}
```

## License

This project is released under the CC BY-NC-ND 4.0 license.
