# BERT+ProtoNet
This is the method performs the best in the previous experiments of cross-domain intent detection, which attempts to do few shot learning with BERT and prototypical network. This is the small paper describing and detailing the approach [paper](https://github.com/arijitx/Fewshot-Learning-with-BERT/raw/master/FEW_SHOT_INTENT_CLASSIFICATION_BERT.pdf). 

Note: I have changed the way of training and testing:
In the original setting, one support utterance and one query utterance for each intent are sampled randomly every step. 
In my setting, all the utterances are divided into batches as query set. In each batch, each intent sample one utterance randomly as support set.

## Dependencies

    * pytorch == 3.6.5
    * torch == 1.0.0
    * re == 2.2.1
    * pytorch_pretrained_bert == 0.6.2
    * numpy == 1.15.4
    
## Directory Structure

    ./data: preprocessed data of Lena and Moli
    data.py: read data
    model.py: the Bert_ProtoNet model
    train.py: train and test procedure
    run.sh: start to train

## Data Schema

    ./data/lena/seq.in  -> 'utterance\n'
    ./data/lena/label   -> 'label\n'
    ./data/moli/seq.in  -> 'utterance\n'
    ./data/moli/label   -> 'label\n'

## Quickstart
### Step 1: Download and unzip the pretrained BERT model
   
    1. download the BERT model and the corresponding vocab file to the root path
    2. tar -zxvf bert-base-uncased.tar.gz
    3. put vocab file into model dir
    
### Step 2: Train the model
 
    // if you want to change the data to transfer, please reedit 'train_data' and 'dev_data' in run.sh
    bash run.sh
    
       
## Ackownledge
Thanks for the original repository [arijitx/Fewshot-Learning-with-BERT](https://github.com/arijitx/Fewshot-Learning-with-BERT)
