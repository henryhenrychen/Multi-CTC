## Requirements
- Python 3.7
- Pytorch 1.4
- pytorchaudio
- pandas
- matplotlib
- editdistance
- [PyYaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
- numpy
- [comet_ml](https://www.comet.ml/site/)
- tqdm 
- [phonemizer](https://github.com/bootphon/phonemizer)

## Preprocess
- preprocess English (source language)
    - Modify ```root``` in preprocess/preprocess_libri.py to your path to LibriSpeech 
    - run ```python preprocess/preprocess_libri.py``` and it will generate Englsih meta data in dir ```data```
- preprocess French (target language)
    - Modify ```root``` in preprocess/preprocess_gp.py to your path to GlobalPhone 
    - run ```python preprocess/preprocess_gp.py``` and it will generate French meta data in dir ```data```


## Pre-training
- base configurations are all defined in `config/pretrain/en.yaml`
- run ```python main --config config/pretrain/en.yaml --save_every ```
    - it automatically save checkpoints in dir ```ckpt``` with sub-dir named after ```en.yaml``` as ```en_sd0```. 
    - checkpoints contain best-performed ```best_per.pth``` and every steps 

## Adaptation / Finetuning
- base configuratoins are defined in ```config/adaptation/adapt_ipa_fr.yaml```
    - if you want the settings with output unit = chatacter, change ```corpus['target']``` from ```ipa``` to ```char``` and ```corpus['vocab_file']``` from ```corpus/fr.ipa.txt``` to ```corpus/fr.txt.txt```
    - we have 3 methos on output embedding when doing transfer learning 
        - ipa: use ipa as the ground truth to transfer, set ```transfer['method']``` as ```ipa``` （專家知識方法）
        - no: random intialize output embeddings, set ```transfer['method']``` as ```no```（從頭學習方法）
        - mapping: use learned mapping to initialize output embedding, set ```transfer['method']``` as ```mapping``` （對映學習方法）
- run ```python scripts/run_adapt.py --pretrain_path ckpt/en_sd0/ --pretrain_config config/pretrain/en.yaml --adapt_config config/adaptation/adapt_ipa_fr.yaml``` 
    - it automatically picks top 20 best-performed pretrain checkpoints as initialization and generates configs for adapatation according to adaptation 
    - run the commands on stdout 


## Testing 
- base configurations are defined in ```config/test_greedy.yaml```
    - you can change batch_size or beam search decoding size as you want
- run ```python scripts/run_test.py --model_path PATH_TO_CKPT --test_config TEST_CONFIG``` to generate testing configs
    - ex: ```python scripts/run_test.py --model_path ckpt/en_sd0_adapt_ipa_fr/step10000-frac1-lr1/ --test config/test_greedy.yaml```
    - run commands on stdout
    - running results would be saved under ```result``` in ```output.csv```
