from source import extract_words, pick_dump, pick_undump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import Levenshtein as lev
from collections import defaultdict
import re
import pickle

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

def dump_lang(path, model):
    with open(path, mode='wb') as f:
        pickle.dump(model, f)

def undump_lang(path):
    with open(path, mode='rb') as f:
        return pickle.load(f) 

def default_factory_second_help():
    return 1

def default_factory_second() :
    return [0, defaultdict(default_factory_second_help)]

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
                        else :
                            self.model_dict['^' + fix[cur_pos_s]][0] += 1
                        cur_pos_s -= 1
                        self.all_bigrams += 1
        for key, value in self.model_dict.items() :
            cur_sum = 0
            for key1 in value[1].keys() :
                cur_sum += value[1][key1]
            for key1 in value[1].keys() :
                value[1][key1] = 1 - value[1][key1] / cur_sum
            value[1][key] = 0
        
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
                        self.model_dict['^_'][1]['^' + orig[dpos]] = tmp
                    lev_sum += tmp
                elif op == 'delete' :
                    tmp = self.model_dict['^' + fix[spos]][1]['^_']
                    if tmp == 0:
                        tmp = self.smooth_const
                        self.model_dict['^' + fix[spos]][1]['^_'] = tmp
                    lev_sum += tmp
                else : # 'replace'
                    tmp = self.model_dict['^' + fix[spos]][1]['^' + orig[dpos]]
                    if tmp == 0:
                        tmp = self.smooth_const
                        self.model_dict['^' + fix[spos]][1]['^' + orig[dpos]] = tmp
                    lev_sum += tmp
            else : # for all cases
                if op == 'insert' :
                    tmp = self.model_dict[fix[spos - 1] + '_'][1][fix[spos - 1] + orig[dpos]]
                    if tmp == 0:
                        tmp = self.smooth_const
                        self.model_dict[fix[spos - 1] + '_'][1][fix[spos - 1] + orig[dpos]] = tmp
                    lev_sum += tmp
                elif op == 'delete' :
                    tmp = self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + '_']
                    if tmp == 0:
                        tmp = self.smooth_const
                        self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + '_'] = tmp
                    lev_sum += tmp
                else : # 'replace'
                    tmp = self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + orig[dpos]]
                    if tmp == 0 :
                        tmp = self.smooth_const
                        self.model_dict[fix[spos - 1] + fix[spos]][1][fix[spos - 1] + orig[dpos]] = tmp
                    lev_sum += tmp
            cur_pos_s -= 1
        return self.a ** (-lev_sum)

def dump_err(path, model):
    with open(path, mode='wb') as f:
        pickle.dump(model, f)

def undump_err(path):
    with open(path, mode='rb') as f:
        return pickle.load(f) 

need_fix = []
no_need_fix = []

with open('./queries_all.txt', encoding='utf-8') as f:
    for line in f:
        line = re.sub('\n', '', line.lower()).split('\t')
        if len(line) == 2:
            need_fix.append(line[0])
            no_need_fix.append(line[1])
        else:
            no_need_fix.append(line[0])

all_queries = need_fix + no_need_fix
len_all_queries = len(all_queries)

# model = undump_lang('./lang_model')
model = Language_model()
model.fit('./queries_all.txt')
err_model = Error_model()
err_model.fit('./queries_all.txt')
# tfidf = TfidfVectorizer()
# tfidf.fit(all_queries)
# tfidf = pick_undump('./tfidf')
# all_tfidf = tfidf.transform(all_queries)
# svd = TruncatedSVD(n_components=15)
# svd.fit(all_tfidf)

queries = []

with open('./queries_all.txt', encoding='utf-8') as f:
    for line in f:
        queries.append(re.sub('\n','',line))

layout = {'a' : 'ф','b' : 'и','c' : 'с','d' : 'в','e' : 'у','f' : 'а','g' : 'п','h' : 'р','i' : 'ш','j' : 'о','k' : 'л','l' : 'д','m' : 'ь','n' : 'т','o' : 'щ','p' : 'з','q' : 'й','r' : 'к','s' : 'ы','t' : 'е','u' : 'г','v' : 'м','w' : 'ц','x' : 'ч','y' : 'н','z' : 'я','`' : 'ё','[' : 'х',']' : 'ъ',';' : 'ж','\'' : 'э',',' : 'б','.' : 'ю','@' : '"','#' : '№','$' : ';','^' : ':','&' : '?','/' : '.','?' : ',','{' : 'Х','}' : 'Ъ','|' : '/','~' : 'Ё','\\' : '\\','1' : '1','2' : '2','3' : '3','4' : '4','5' : '5','6' : '6','7' : '7','8' : '8','9' : '9','0' : '0','(' : '(',')' : ')','*' : '*','%' : '%','!' : '!','-' : '-','=' : '=','_' : '_','+' : '+','A' : 'Ф','B' : 'И','C' : 'С','D' : 'В','E' : 'У','F' : 'А','G' : 'П','H' : 'Р','I' : 'Ш','J' : 'О','K' : 'Л','L' : 'Д','M' : 'Ь','N' : 'Т','O' : 'Щ','P' : 'З','Q' : 'Й','R' : 'К','S' : 'Ы','T' : 'Е','U' : 'Г','V' : 'М','W' : 'Ц','X' : 'Ч','Y' : 'Н','Z' : 'Я',':' : 'Ж','"' : 'Э','<' : 'Б','>' : 'Ю', ' ' : ' '}
inv_layout = {'ф' : 'a','и' : 'b','с' : 'c','в' : 'd','у' : 'e','а' : 'f','п' : 'g','р' : 'h','ш' : 'i','о' : 'j','л' : 'k','д' : 'l','ь' : 'm','т' : 'n','щ' : 'o','з' : 'p','й' : 'q','к' : 'r','ы' : 's','е' : 't','г' : 'u','м' : 'v','ц' : 'w','ч' : 'x','н' : 'y','я' : 'z','ё' : '`','х' : '[','ъ' : ']','ж' : ';','э' : '\'','б' : ',','ю' : '.','"' : '@','№' : '#',';' : '$',':' : '^','?' : '&','.' : '/',',' : '?','Х' : '{','Ъ' : '}','/' : '|','Ё' : '~','\\' : '\\','1' : '1','2' : '2','3' : '3','4' : '4','5' : '5','6' : '6','7' : '7','8' : '8','9' : '9','0' : '0','(' : '(',')' : ')','*' : '*','%' : '%','!' : '!','-' : '-','=' : '=','_' : '_','+' : '+','Ф' : 'A','И' : 'B','С' : 'C','В' : 'D','У' : 'E','А' : 'F','П' : 'G','Р' : 'H','Ш' : 'I','О' : 'J','Л' : 'K','Д' : 'L','Ь' : 'M','Т' : 'N','Щ' : 'O','З' : 'P','Й' : 'Q','К' : 'R','Ы' : 'S','Е' : 'T','Г' : 'U','М' : 'V','Ц' : 'W','Ч' : 'X','Н' : 'Y','Я' : 'Z','Ж' : ':','Э' : '"','Б' : '<','Ю' : '>', ' ' : ' '}

def is_english(s):
        '''
        takes only lower words
        '''
        is_ok = re.search(r'\b[a-z0-9_]+\b', s, flags=re.IGNORECASE) is not None
        return is_ok

def is_russian(s):
    '''
    takes only lower words
    '''
    is_ok = re.search(r'\b[а-я0-9_]+\b', s) is not None
    return is_ok

def change_layout(query): # do smth with non russian/ukrainian words
    res = ''
    lang = None
    if is_english(query):
        for char in query:
            try:
                res += layout[char]
            except KeyError:
                res += char
        lang = 0 # for english
    elif is_russian(query):
        for char in query:
            try:
                res += inv_layout[char]
            except KeyError:
                res += char
        lang = 1 # for russian
    else:
        return query, False, lang
    return res, True, lang

eng_to_rus = {}
rus_to_eng = {}
for query in queries:
    query = query.lower().split('\t')
    query = query[len(query)//2]
    for word in extract_words(query):
        if is_english(word):
            ch_l = change_layout(word)[0]
            eng_to_rus[word] = ch_l
            rus_to_eng[ch_l] = word
        elif is_russian(word):
            ch_l = change_layout(word)[0]
            rus_to_eng[word] = ch_l
            eng_to_rus[ch_l] = word

pick_dump('./eng_to_rus', eng_to_rus)
pick_dump('./rus_to_eng', rus_to_eng)
pick_dump('./err_model', err_model)
pick_dump('./lang_model', model)