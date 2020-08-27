# BERT+ProtoNet
This is the method performs the best in the previous experiments of cross-domain intent detection, which attempts to do few shot learning with BERT and prototypical network. This is the small paper describing and detailing the approach [paper](https://github.com/arijitx/Fewshot-Learning-with-BERT/raw/master/FEW_SHOT_INTENT_CLASSIFICATION_BERT.pdf). 

I have changed the way of training and testing:
In the original setting, one support utterance and one query utterance for each intent are sampled randomly every step. 
In my setting, all the utterances are divided into batches as query set. In each batch, each intent sample one utterance randomly as support set.

## Directory Structure:

    ./data: lena&moli data
    data.py: read data
    model.py: the model
    train.py: train and test procedure
    run.sh: start to train and test

## Data Schema:

    ./data/lena/seq.in  -> 'utterance\n'
    ./data/lena/label   -> 'label\n'
    ./data/moli/seq.in  -> 'utterance\n'
    ./data/moli/label   -> 'label\n'

## Quickstart:
### Step 1: Download and unzip the pretrained BERT model
   
    1. download the BERT model and the corresponding vocab file to the root path
    2. tar -zxvf bert-base-uncased.tar.gz
    3. put vocab file into model dir
    
### Step 2: Train the model
 
    bash run.sh
       
## Ackownledge: 
Thanks for the original repository [arijitx/Fewshot-Learning-with-BERT](https://github.com/arijitx/Fewshot-Learning-with-BERT)
