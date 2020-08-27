import torch
from torch import nn 
import numpy as np 

from data import IntentDset
from pytorch_pretrained_bert.modeling import BertModel

class ProtNet(nn.Module):
	def __init__(self, n_input = 768, n_output = 128, bert_model = 'bert-base-uncased'):
		super(ProtNet,self).__init__()
		self.bert = BertModel.from_pretrained('../Fewshot-Learning-with-BERT-master/bert-base-uncased')

	def forward(self, input_ids, input_mask):
		all_hidden_layers,_ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask)
		hn = all_hidden_layers[-1]
		cls_hn = hn[:,0,:]
		return cls_hn

