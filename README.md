# BERT+ProtoNet
This is the method performs the best in the previous experiments of cross-domain intent detection, which attempts to do few shot learning with BERT and prototypical network. 

This is the small paper describing and detailing the approach [paper](https://github.com/arijitx/Fewshot-Learning-with-BERT/raw/master/FEW_SHOT_INTENT_CLASSIFICATION_BERT.pdf)

Directory Structure:

    ./data: lena&moli data
    data.py: read data
    model.py: the model
    train.py: train and test model

Data Schema:

    ./data/lena/seq.in  -> 'utterance\n'
    ./data/lena/label   -> 'label\n'
    ./data/moli/seq.in  -> 'utterance\n'
    ./data/moli/label   -> 'label\n'

Quickstart:
Step 1: Download and unzip the pretrained BERT model
   
    1. download the BERT model and the corresponding vocab file to the root path
    2. tar -zxvf bert-base-uncased.tar.gz
    3. put vocab file into model dir
    
Step 2: Train the model
 
    bash run.sh
        
