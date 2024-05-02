from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from utils.bleu.bleu import Bleu
import numpy as np
import json
import torch

def matching_token_num(pred, gold):
    unique_pred = set(pred)
    unique_gold = set(gold)
    
    matching_token = unique_pred.intersection(unique_gold)
    
    return len(matching_token)

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class NLPEvaluator(object):
    def __init__(self, prediction, verbose=False):
        # if not prediction_filename:
        #     raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.prediction = prediction#self.import_prediction(prediction_filename)

        self.tokenizer = PTBTokenizer()
        self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
            ]
        # if self.verbose:
        #     self.bertscorer = (BertScore(), "BertScore")

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("Loading submission...")
        submission = json.load(open(prediction_filename))
        results = {}
        for vid_id in submission:
            results[vid_id] = submission[vid_id]
        return results

    def evaluate(self):
        self.scores = {}
        scores = self.example_evaluate(self.prediction)
        for metric, score in scores.items():
            if metric not in self.scores:
                self.scores[metric] = []
            self.scores[metric].append(score)
        return self.scores
        

    def example_evaluate(self, prediction):
        unique_index = 0
        cur_res = {}
        cur_gts = {}

        for pred in prediction:
            cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
            cur_gts[unique_index] = [{'caption': remove_nonascii(pred['gt_sentence'])}]
            unique_index += 1 

        all_scores = {}
        tokenize_res = self.tokenizer.tokenize(cur_res)
        tokenize_gts = self.tokenizer.tokenize(cur_gts)
        output = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print('computing %s score...'%(scorer.method()))

            kargs = {'gts':tokenize_gts, 'res':tokenize_res}
            score, scores = scorer.compute_score(**kargs)

            if type(method) == list: 
                for sc, scs, m in zip(score, scores, method):
                    output[m] = float(sc)
                    if self.verbose: 
                        print("Calculated %s: %0.5f"%(m, sc))
            else:
                output[method] = np.mean(list(scores))
                if self.verbose: 
                    print("Calculated %s: %0.3f" % (method, output[method]))

        # if self.verbose: 
        #     scorer, method = self.bertscorer
        #     kargs = {'gts':gts, 'res':res}
        #     score, scores = scorer.compute_score(**kargs)
        #     output[method] = score 
        
        return output 