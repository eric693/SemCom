# Semantic Communication (SemCom)

<img width="780" alt="nr_protocol" src="https://github.com/eric693/SemCom/assets/75469714/667e902e-73a0-4d73-bfdf-cadbc2f6a765">

<img width="780" alt="convertional_communication" src="https://github.com/eric693/SemCom/assets/75469714/b17406c4-7925-48f2-a8d6-edaa7348ab71">

<img width="781" alt="semantic_communication" src="https://github.com/eric693/SemCom/assets/75469714/8fa9ce52-2da3-4df7-9aa1-1b27e8484ed2">

<img width="781" alt="semantic_communication" src="https://github.com/eric693/SemCom/assets/75469714/9a01084d-4839-47ef-9497-6d9225b6465f">

<img width="780" alt="OAI-flow-chart" src="https://github.com/eric693/SemCom/assets/75469714/99b33fcc-38e9-47e2-903d-ad08021321b1">


**SemCom is designed to preserve the semantic information instead of strictly securing the bit-level precision. It enables a general-purpose, large-scale, wireless, and semantic communication framework.**

## Features
+ A schematic shift from bit-precision to semantic consistency.
+ Compatible with any (non-differenable) semantic similarity metric as the objective function.
+ RL-based end-to-end optimization on non-differentiable and unknown wireless channel with high-dimensional action/semantic space.


## Requirements
```
pip3 install -r requirements.txt
```


## Dataset Preparation
```
cd $your_data_root
wget https://www.statmt.org/europarl/v7/fr-en.tgz
tar -zxvf fr-en.tgz   
python3 preprocess_captions.py --data_root Europarl/
sudo cp english.pkl Europarl/
sudo cp english_vocab.pkl Europarl/
```

## Training

### Training SemanticRL-JSCC

```
# AWGN-CE-Stage1
python3 Trainng_SemanticRL.py --training_config ./config/config_AWGN_CE.yaml --dataset_path Europarl/

# AWGN-CE-Stage2
python3 Trainng_SemanticRL.py --training_config ./config/config_AWGN_CE_Stage2.yaml --dataset_path Europarl/

# AWGN-RL-Stage2
python3 Trainng_SemanticRL.py --training_config ./config/config_AWGN_RL.yaml --dataset_path Europarl/
```

You can change the type of random channel to trian and test in different scenarios. For more details, run `python Trainng_SemanticRL.py --help`.


### Training SemanticRL-SCSIU
```
python Trainng_SemanticRL.py --training_config ./config/config_AWGN_RL_SCSIU.yaml --dataset_path $your_data_root
```

## Inference

Download the pretrained model. Place them into the root directory.

[Baidu Netdisk](https://pan.baidu.com/s/1wJ8ZFXyGugnqK1r_DhDkCw) extraction code: `fp3t`
 

```
SemCom
├── ckpt_AWGN_CE_Stage2
│   └── all checkpoints
├── ckpt_AWGN_RL 			  (the second version, see arXiv. trained with shuffle=True)
├── ckpt_AWGN_RL_SemanticRLv1 (the first version, see arXiv. trained with shuffle=False)
├── Evaluation
│   └── Inference_Given_Input.py
│   └── Run_Inference_Checkpoints.py
│   └── ...
├── Trainng_SemanticRL.py
├── ...
```	

**Reproducing Quantitative Results**

```
# Step1. load checkpoint and run inference on test set. 
# Output dir: ./Evaluation/InferenceResutls/$EXP_NAME/xxx.json (output tokens inside)
python Evaluation/Run_Inference_Checkpoints.py --path $CKPT_PATH --name $EXP_NAME --data_root $your_data_root
# Step2. calculate metrics like BLEU, CIDER etc.
# Output dir: ./Evaluation/EvalResults/$EXP_NAME/xxx.json (scores inside)
python Evaluation/CaptionMetrics-master/Eval_Metric_Checkpoints.py --path Evaluation/InferenceResutls/$EXP_NAME --name $EXP_NAME
```

**Reproducing Visual Results**

```
# Output dir: std output (i.e., your screen) (sentences of type str)
python Evaluation/Inference_Given_Input.py
```

Your trained model may behave a little different from ours, but they should be similar.


## Integrating SemanticRL with your own framework

Besides `LSTM` backbone, we provide a `Transformer` backbone to facilitate further researches. You can rewrite methods in `model.py` to customize your own framework. SemanticRL is model-agnostic. You may also design any semantic similarity metric to build a customed communication system.

## Design of generative AI




