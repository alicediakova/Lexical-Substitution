#!/usr/bin/env python
import sys
import string

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    
    res = set()
    lemlist = wn.lemmas(lemma, pos = pos)
    for lem in lemlist:
        synlems = lem.synset().lemmas()
        for sl in synlems:
            name = sl.name()
            if '_' in name:
                name = name.replace('_',' ')
            res.add(name)
    
    if lemma in res:
        res.remove(lemma)
    
    return list(res) #according to ED post #526 we were instructed to return a list even though the hw description says to return a set

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    lemma = context.lemma
    pos = context.pos
    
    counts_dict = dict()
    
    lemlist = wn.lemmas(lemma, pos = pos)
    for lem in lemlist:
        synlems = lem.synset().lemmas()
        for sl in synlems:
            name = sl.name()
            if name == lemma: #skip if same as input word
                continue
            count = sl.count()
            if name in counts_dict.keys():
                counts_dict[name] += count
            else:
                counts_dict[name] = count
                
    #find candidate with max count
    max_count = -1
    res_candidate = ""
    for k,v in counts_dict.items():
        if v > max_count:
            max_count = v
            res_candidate = k

    return res_candidate

def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3
    lemma = context.lemma
    pos = context.pos

    synsets_set = set() #all possible synsets the target word appears in
    lems = wn.lemmas(lemma, pos = pos)
    for l in lems:
        synsets_set.add(l.synset())
    synsets = list(synsets_set)

    # 1: group 1: combine left and right contexts into one list
    context_ = context.left_context + context.right_context
    
    # 2: remove stopwords from context list
    group1 = list()
    stop_words = stopwords.words('english')
    for word in context_:
        if word not in stop_words:
            group1.append(word)

    overlap_dict = dict() #keys are synsets, values are overlap values
    for synset in synsets:
        # 3: group 2: tokenize, normalize, then combine def and examples
        group2 = tokenize(synset.definition()) # synset definition
        for examp in synset.examples():
            group2 = group2 + tokenize(examp)
            
        # 4: for each hypernym: tokenize and normalize defs and examples
            # add these lists to group 2
        for hyp in synset.hypernyms():
            group2 = group2 + tokenize(hyp.definition())
            for examp in hyp.examples():
                group2 = group2 + tokenize(examp)
        
        # calculate and store overlap between the groups in dict
        overlap = len(set(group1)&set(group2)) # set intersection of the groups
        overlap_dict[synset] = overlap

    # determine the synset with the highest overlap
    max_overlap = -1
    ho_synset = synsets[0]
    for k,v in overlap_dict.items():
        if v > max_overlap:
            max_overlap = v
            ho_synset = k
            
    overlap_tie = False
    ho_synset_count = 0
    for k,v in overlap_dict.items():
        if v == max_overlap:
            ho_synset_count += 1
            
    if ho_synset_count > 1:
        overlap_tie = True

    if overlap_tie:
        # if tie in overlap:
            # return most_freq_synon of synset where target word has highest .count()
        return most_freq_synset(lemma, pos)
    else:
        # if no tie in overlap:
            # return most_freq_synon of the synset with the highest overlap
        return most_freq_synon(ho_synset)
    

def part6_predictor(context : Context) -> str: # same as lesk but also takes hyponyms into account (shows better prediction scores than lesk algorithm)
    lemma = context.lemma
    pos = context.pos

    synsets_set = set() #all possible synsets the target word appears in
    lems = wn.lemmas(lemma, pos = pos)
    for l in lems:
        synsets_set.add(l.synset())
    synsets = list(synsets_set)

    # 1: group 1: combine left and right contexts into one list
    context_ = context.left_context + context.right_context
    
    # 2: remove stopwords from context list
    group1 = list()
    stop_words = stopwords.words('english')
    for word in context_:
        if word not in stop_words:
            group1.append(word)

    overlap_dict = dict() #keys are synsets, values are overlap values
    for synset in synsets:
        # 3: group 2: tokenize, normalize, then combine def and examples
        group2 = tokenize(synset.definition()) # synset definition
        for examp in synset.examples():
            group2 = group2 + tokenize(examp)
            
        # 4: for each hypernym: tokenize and normalize defs and examples
            # add these lists to group 2
        for hyper in synset.hypernyms():
            group2 = group2 + tokenize(hyper.definition())
            for examp in hyper.examples():
                group2 = group2 + tokenize(examp)
                
        # 5: for each hyponym: tokenize and normalize defs and examples
            # add these lists to group 2
        for hypo in synset.hyponyms():
            group2 = group2 + tokenize(hypo.definition())
            for examp in hypo.examples():
                group2 = group2 + tokenize(examp)
        
        # calculate and store overlap between the groups in dict
        overlap = len(set(group1)&set(group2)) # set intersection of the groups
        overlap_dict[synset] = overlap

    # determine the synset with the highest overlap
    max_overlap = -1
    ho_synset = synsets[0]
    for k,v in overlap_dict.items():
        if v > max_overlap:
            max_overlap = v
            ho_synset = k
            
    overlap_tie = False
    ho_synset_count = 0
    for k,v in overlap_dict.items():
        if v == max_overlap:
            ho_synset_count += 1
            
    if ho_synset_count > 1:
        overlap_tie = True

    if overlap_tie:
        # if tie in overlap:
            # return most_freq_synon of synset where target word has highest .count()
        return most_freq_synset(lemma, pos)
    else:
        # if no tie in overlap:
            # return most_freq_synon of the synset with the highest overlap
        return most_freq_synon(ho_synset)
    
    
def most_freq_synon(synset) -> str: #lesk helper: takes in a synset and returns the most frequent synonym according to WordNet
    
    counts_dict = dict()
    
    synlems = synset.lemmas()
    for sl in synlems:
        name = sl.name()
        count = sl.count()
        if name in counts_dict.keys():
            counts_dict[name] += count
        else:
            counts_dict[name] = count
   
    #find candidate with max count
    max_count = -1
    res_candidate = ""
    for k,v in counts_dict.items():
        #print(k,v)
        if v > max_count:
            max_count = v
            res_candidate = k

    return res_candidate
    
def most_freq_synset(lemma, pos) -> str: #lesk helper: called on ties, selects most frequent synset and returns its most frequent lexeme
    
    counts_dict = dict() # keys are synsets, values are total counts
    lemmas = wn.lemmas(lemma, pos = pos)
    for lem in lemmas:
        synset = lem.synset()
        syn_lems = synset.lemmas()
        for sl in syn_lems:
            if synset in counts_dict.keys():
                counts_dict[synset] += sl.count()
            else:
                counts_dict[synset] = sl.count()
                
    max_count = -1
    mf_synset = lemmas[0].synset()
    for k,v in counts_dict.items():
        #print(k,v)
        if v > max_count:
            max_count = v
            mf_synset = k
    
    return most_freq_synon(mf_synset)
        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context : Context) -> str:
        # Part 4
        lemma = context.lemma
        pos = context.pos
        
        similarity_dict = dict() #keys are synonyms, values are similarity value to lemma
        last = ""
        
        lemlist = wn.lemmas(lemma, pos = pos)
        for lem in lemlist:
            synlems = lem.synset().lemmas()
            for sl in synlems:
                name = sl.name()
                last = name
                if name != lemma and name in self.model.key_to_index:
                    #print(lemma, name)
                    curr_sim = self.model.similarity(lemma, name)
                    if name in similarity_dict.keys():
                        if curr_sim > similarity_dict[name]:
                            similarity_dict[name] = curr_sim
                    else:
                        similarity_dict[name] = curr_sim
#                if '_' in name:
#                    name = name.replace('_',' ')
#                res.add(name)
        
        max_sim = 0
        most_similar = last
        for k,v in similarity_dict.items():
            #print(k,v)
            if v > max_sim:
                max_sim = v
                most_similar = k
        
        return most_similar
        

class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Part 5
        
        lemma = context.lemma
        pos = context.pos
        
        cands = get_candidates(lemma, pos) # 1: obtain set of candidate synonyms
        index = len(context.left_context) #store index of target word
        
        cntxt = " ".join(context.left_context) + " [MASK] " + " ".join(context.right_context) # 2: convert info in context into suitable masked input represention for DistilBERT model
        
        # 3: run DistilBERT model on input representation
        input_toks = self.tokenizer.encode(cntxt)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][index])[::-1] # sort in increasing order
        res = self.tokenizer.convert_ids_to_tokens(best_words)
    
        # 4: select, from candidates, the highest scoring word
        for word in res: #cross check best words with the candidate synonyms
            if word in cands: # the first word from best_words that overlaps with cands has the highest score since res is in highest-to-lowest order
                return word
    
        return "" #none of the candidate words are in the best words

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    #predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = part6_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = wn_frequency_predictor(context)
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor.predict(context)
        
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    
    
    
        
    
