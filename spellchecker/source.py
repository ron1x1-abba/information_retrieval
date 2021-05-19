import re
import pickle
from collections import defaultdict
import Levenshtein as lev

def extract_words(query) :
    '''
    May add "", :, /, bool operations(&, !, |)
    '''
    res = query.lower()
    res = re.sub(r'_', ' ', res)
#     res = re.sub(r'[^\w\s-]', ' ', query.lower()) # delete symbols that we won't use 100%
#     res = re.findall(r'(\b\w+-\w+\b | \b\w+\b)', res)
#     res = [word for word in res.split(' ') if word != '']
    res = re.findall(r'\w+', res)
    return res

def extract_non_words(query):
    return re.findall(r'[^\w]+', query)

def extract_words_register(query):
    res = re.sub(r'_', ' ', query)
#     res = re.sub(r'[^\w\s-]', ' ', query.lower()) # delete symbols that we won't use 100%
#     res = re.findall(r'(\b\w+-\w+\b | \b\w+\b)', res)
#     res = [word for word in res.split(' ') if word != '']
    res = re.findall(r'\w+', res)
    return res

def extract_features(query):
    cur = []
    words = extract_words(query)
    cur.append(lang_model.query_prob(query)) # probability of this query in lang model
    cur.append(len(query)) # length of query
    cur.append(len(words)) # amount of words
    cur.append(len(re.findall(r'[^\w]', query)))# amount of non word symbols
    amount_in_voc = 0
    for word in words:# amount of words in vocabulary
        if lang_model[word][0] != lang_model.smooth_const:
            amount_in_voc += 1
    cur.append(amount_in_voc)
    return cur

def dump_err(path, model):
    with open(path, mode='wb') as f:
        pickle.dump(model, f)

def undump_err(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)

class Error_model :
    def __init__(self, laplace=1, a=1.2) :
        self.model_dict = defaultdict(default_factory_second)
        self.laplace_const = laplace
        self.smooth_const = 1
        self.all_bigrams = 0
        self.a = a

    def __getitem__(self, key) : # add laplace smoothing
        res = self.model_dict[key]
        if res[0] == 0 :
            res[0] = self.smooth_const
        return res
        
    def __setitem__(self, key, value) :
        self.model_dict[key] = value

    def fit(self, path) : # add different types of corrections : split, join and layout
        with open(path, mode='r', encoding='utf-8') as f :
            for line in f:
                line = re.sub(r'\n', '', line)
                line = line.split('\t')
                if len(line) != 2 : # in case of correct query
                    continue
                orig = line[0].lower()
                fix = line[1].lower()
                orig_words = extract_words(orig)
                fix_words = extract_words(fix)
                if len(orig_words) != len(fix_words):
                    continue
                for orig, fix in zip(orig_words, fix_words):
                    orig_len = len(orig)
                    fix_len = len(fix)
                    cur_pos_s = fix_len - 1
                    for op, spos, dpos in reversed(lev.editops(fix, orig)) :
                        while cur_pos_s > spos : 
                            self.model_dict[fix[cur_pos_s - 1] + fix[cur_pos_s]][0] += 1
    #                         self.model_dict[fix[cur_pos_s - 1] + fix[cur_pos_s]][1][fix[cur_pos_s - 1] + fix[cur_pos_s]] += 1
                            cur_pos_s -= 1
                            self.all_bigrams += 1
                        if cur_pos_s == 0 :
                            if op == 'insert' :
                                self.model_dict['^_'][0] += 1
                                self.model_dict['^_'][1]['^' + orig[dpos]] += 1
                            elif op == 'delete' :
                                self.model_dict['^' + fix[spos]][0] += 1
                                self.model_dict['^' + fix[spos]][1]['^_'] += 1
                            else : # 'replace'
                                self.model_dict['^' + fix[spos]][0] += 1
                                self.model_dict['^' + fix[spos]][1]['^' + orig[dpos]] += 1
                            self.all_bigrams += 1
                        else : # for all cases
                            if op == 'insert' :
                                self.model_dict[fix[spos - 1] + '_'][0] += 1
                                self.model_dict[fix[spos - 1] + '_'][1][fix[spos - 1] + orig[dpos]] += 1
                            elif op == 'delete' :
                                self.model_dict[fix[spos - 1] + fix[spos]][0] += 1
                                self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + '_'] += 1
                            else : # 'replace'
                                self.model_dict[fix[spos - 1] + fix[spos]][0] += 1
                                self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + orig[dpos]] += 1
                        cur_pos_s -= 1
                        self.all_bigrams += 1
                    while cur_pos_s >= 0:
                        if cur_pos_s != 0:
                            self.model_dict[fix[cur_pos_s - 1] + fix[cur_pos_s]][0] += 1
    #                         self.model_dict[fix[cur_pos_s - 1] + fix[cur_pos_s]][1][fix[cur_pos_s - 1] + fix[cur_pos_s]] += 1
                        else :
                            self.model_dict['^' + fix[cur_pos_s]][0] += 1
    #                         self.model_dict['^' + fix[cur_pos_s]][1]['^' + fix[cur_pos_s]] += 1
                        cur_pos_s -= 1
                        self.all_bigrams += 1
        for key, value in self.model_dict.items() :
            cur_sum = 0
            for key1 in value[1].keys() :
                cur_sum += value[1][key1]
#                 value[1][key1] /= value[0]
                # value[1][key1] = (value[0] - value[1][key1]) / value[0]
            for key1 in value[1].keys() :
                value[1][key1] = 1 - value[1][key1] / cur_sum
            value[1][key] = 0
#         self.smooth_const = 1 / self.all_bigrams
        
    def proba(self, orig, fix, a=2) :
        lev_sum = 0
        fix_len = len(fix)
        orig = orig.lower()
        fix = fix.lower()
        cur_pos_s = fix_len - 1
        for op, spos, dpos in reversed(lev.editops(fix, orig)) :
            while cur_pos_s > spos :
                cur_pos_s -= 1
            if cur_pos_s == 0 :
                if op == 'insert' :
                    tmp = self.model_dict['^_'][1]['^' + orig[dpos]]
                    if tmp == 0 :
                        tmp = self.smooth_const
#                         tmp = 1 / self.model_dict['^_'][0]
                        self.model_dict['^_'][1]['^' + orig[dpos]] = tmp
                    lev_sum += tmp
                elif op == 'delete' :
                    tmp = self.model_dict['^' + fix[spos]][1]['^_']
                    if tmp == 0:
                        tmp = self.smooth_const
#                         tmp = 1 / self.model_dict['^' + fix[spos]][0]
                        self.model_dict['^' + fix[spos]][1]['^_'] = tmp
                    lev_sum += tmp
                else : # 'replace'
                    tmp = self.model_dict['^' + fix[spos]][1]['^' + orig[dpos]]
                    if tmp == 0:
                        tmp = self.smooth_const
#                         tmp = 1 / self.model_dict['^' + fix[spos]][0]
                        self.model_dict['^' + fix[spos]][1]['^' + orig[dpos]] = tmp
                    lev_sum += tmp
            else : # for all cases
                if op == 'insert' :
                    tmp = self.model_dict[fix[spos - 1] + '_'][1][fix[spos - 1] + orig[dpos]]
                    if tmp == 0:
                        tmp = self.smooth_const
#                         tmp = 1 / self.model_dict[fix[spos - 1] + '_'][0]
                        self.model_dict[fix[spos - 1] + '_'][1][fix[spos - 1] + orig[dpos]] = tmp
                    lev_sum += tmp
                elif op == 'delete' :
                    tmp = self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + '_']
                    if tmp == 0:
                        tmp = self.smooth_const
#                         tmp = 1 / self.model_dict[fix[spos - 1] + fix[spos]][0]
                        self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + '_'] = tmp
                    lev_sum += tmp
                else : # 'replace'
                    tmp = self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + orig[dpos]]
                    if tmp == 0 :
                        tmp = self.smooth_const
#                         tmp = 1 / self.model_dict[fix[spos - 1] + fix[spos]][0]
                        self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + orig[dpos]] = tmp
                    lev_sum += tmp
            cur_pos_s -= 1
        return self.a ** (-lev_sum)
#         return lev_sum

def dump_lang(path, model):
    with open(path, mode='wb') as f:
        pickle.dump(model, f)

def undump_lang(path):
    with open(path, mode='rb') as f:
        return pickle.load(f) 

def pick_dump(path, obj):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)

def pick_undump(path):
    with open(path, mode='rb') as f:
        return pickle.load(f) 

def default_factory_second_help():
    return 1

def default_factory_second() :
    return [0, defaultdict(default_factory_second_help)]

def factory_first_level() :
    return [0 , defaultdict(float)] # count of this word in all queries, next word in bigramms, which starts with this word

class Language_model :
    def __init__(self, laplace=1) :
        self.model_dict = defaultdict(factory_first_level)
        self.laplace_const = laplace
        self.all_words = 1
        self.smooth_const = 1
        
    def __getitem__(self, key) : # add laplace smoothing
        res = self.model_dict[key]
        if res[0] == 0 :
            res[0] = self.smooth_const
        return res
        
    def __setitem__(self, key, value) :
        self.model_dict[key] = value
        
    def fit(self, path) :
        all_words = 0
        with open(path, mode='r', encoding='utf-8') as f :
            for line in f:
                line = line.split('\t')
                line = line[len(line) // 2] # if there is correction in query
                words = extract_words(line)
                words_len = len(words)
                for pos in range(words_len - 1) :
                    all_words += 1
                    elem = self.model_dict[words[pos]]
                    elem[0] += 1 # increase count of this word in all queries
                    elem[1][words[pos + 1]] += 1 #increase count of bigramm in queries
                    self.model_dict[words[pos]] = elem
                if words_len > 0:
                    all_words += 1
                    self.model_dict[words[-1]][0] += 1
        for key, value in self.model_dict.items() :
            for key1 in value[1].keys() :
                value[1][key1] /= self.model_dict[key1][0]
        for key, value in self.model_dict.items() :
            value[0] /= all_words
            self.model_dict[key] = value
        self.all_words = all_words
        self.smooth_const = 1 / all_words
    
    def query_prob(self, query) :
        res = 1
        words = extract_words(query)
        words_len = len(words)
        for pos in range(words_len - 1) :
            bi_prob = self.model_dict[words[pos]][1][words[pos + 1]]
            if bi_prob == 0 : # add laplace smoothing instead
                bi_prob = self.smooth_const
                self.model_dict[words[pos]][1][words[pos + 1]] = bi_prob
            res *= bi_prob
        if words_len > 0 :
            word_prob = self.model_dict[words[-1]][0]
            if word_prob == 0: # add laplace smoothing instead
                word_prob = self.smooth_const
                self.model_dict[words[-1]][0] = word_prob
            res *= word_prob
        else : 
            res = 0 # proba for empty query
        return res

def undump_svd(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)
    
def undump_tfidf(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)