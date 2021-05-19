from source import Language_model, Error_model, extract_words, extract_words_register, pick_dump, pick_undump, \
    default_factory_second, default_factory_second_help, factory_first_level, extract_features, extract_non_words
from queue import PriorityQueue, Empty
from math import log2, log
from collections import defaultdict
import catboost as cat
import numpy as np
from source import extract_words_register
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, accuracy_score
import pickle
import sys
import re

def default_factory_third():
    return [0, defaultdict(list)]

class Node():
    def __init__(self):
        self.nodes = {}
        self.count = 0
        self.prob = 1e-18
        self.all_count = 0

    def add(self, letter):
        if letter not in self.nodes:
            self.nodes[letter] = Node()

class Trie():
    def __init__(self):
        self.length = 0
        self.head = Node()

    def add(self, word):
        pos = 0
        cur = self.head
        if len(word) != 0 :
            cur.all_count += 1
            if '^' + word[0] in cur.nodes:
                cur = cur.nodes['^' + word[0]]
            else:
                cur.add('^' + word[0])
                cur.nodes = dict(sorted(cur.nodes.items()))
                cur = cur.nodes['^' + word[0]]
            cur.count += 1
            pos = 1
        while(pos <= len(word)):
            if(pos == len(word)):
                return
            pos += 1
            if(word[pos-2] + word[pos-1] in cur.nodes):
                cur = cur.nodes[word[pos-2] + word[pos-1]]
            else:
                cur.add(word[pos-2] + word[pos-1])
                cur.nodes = dict(sorted(cur.nodes.items()))
                cur = cur.nodes[word[pos-2] + word[pos-1]]
            cur.count += 1

class Bor:
    # def __init__(self, error_model, language_model, svd, tfidf, classifier, layout, inv_layout, eng_to_rus, rus_to_eng):
    def __init__(self, error_model, language_model, classifier, layout, inv_layout, eng_to_rus, rus_to_eng):
        self.tree = Trie()
        self.err_model = error_model
        self.lang_model = language_model
        # self.svd = svd
        # self.tfidf = tfidf
        self.boost = classifier
        self.layout = layout
        self.inv_layout = inv_layout
        self.eng_to_rus = eng_to_rus
        self.rus_to_eng = rus_to_eng
        self.queue = PriorityQueue()
        self.shit = 0
        self.prob_const = error_model.a

    def loss_function(self, lev, pref_freq, word_prob, a=0.025, b=1, c=25):
        return a * log2(pref_freq) + b * log2(self.prob_const ** (-lev)) + c * log2(word_prob+1)
        
    def fit(self):
        with open('./queries_all.txt', mode='r', encoding='utf-8') as f :
            for line in f:
                line = line.split('\t')
                line = line[len(line) // 2] # if there is correction in query
                words = extract_words(line)
                for word in words:
                    self.tree.add(word)
        self.tree.head.count = 1.01 # frequnecy in case of deletion at the start of creating a word
    
    def search(self, word, aa=0.025, b=1, c=25, best=3, k=6):
        word = word.lower()
        pos = 0
        self.queue = PriorityQueue()
        res = []
        length = len(word)
        cur_len = 0
        cur = self.tree.head
        weights = []
        count_true = 0
        count_false = 0
        if length < 4:
            a = aa * 0.01
        else:
            a = aa
        if length > 3 and length < 11: # ??? maybe 3?
            for pref in sorted(cur.nodes.keys(), key=lambda x : self.loss_function(self.err_model['^'+word[0]][1][x], cur.nodes[x].count, self.lang_model[x[-1]][0],  a, b, c))[-k:]:
                self.queue.put((-self.loss_function(self.err_model['^'+word[0]][1][pref], cur.nodes[pref].count, self.lang_model[pref[-1]][0], a, b, c), 
                                (1, pref[-1], self.err_model['^'+word[0]][1][pref], cur.nodes[pref]))) # replace
        else:
            return [(0, word)]
        iterations = 0
        less_edge = 0
        while cur_len < best and iterations < 20000:
            iterations += 1
            try:
                tmp = self.queue.get_nowait()
            except Empty:
                break
            if tmp[0] > 0.7 and length < 7: # 0.7 -> 0.2 
                continue
            less_edge = 0 if tmp[0] < 0.32 else less_edge + 1 # give some profit in speed, but reduce accuracy, but speed better
            if less_edge > 5:
                break
            cur_pos, cur_pref, cur_lev, cur = tmp[1]
            if cur_lev > 3.0:
                continue
            last_symb = cur_pref[-1] if len(cur_pref) > 0 else ''
            if cur_pos == length:
                if abs(length - len(cur_pref)) >= 3:
                    continue
                candidate = (self.loss_function(cur_lev, cur.count, self.lang_model[cur_pref][0], a, b, c), cur_pref)
                found = False
                for loss, wordd in res:
                    if wordd == cur_pref:
                        found = True
                        break
                if not found:
                    res.append(candidate)
                    cur_len += 1
                continue
            if last_symb != '':
                for pref in sorted(cur.nodes.keys(), key=lambda x : self.loss_function(cur_lev + self.err_model[last_symb+word[cur_pos]][1][x], cur.nodes[x].count, self.lang_model[cur_pref+x[-1]][0], a, b, c))[-k:]:
                    self.queue.put((-self.loss_function(cur_lev + self.err_model[last_symb+word[cur_pos]][1][pref], cur.nodes[pref].count, self.lang_model[cur_pref+pref[-1]][0], a, b, c), 
                                    (cur_pos+1, cur_pref+pref[-1], cur_lev + self.err_model[last_symb+word[cur_pos]][1][pref], cur.nodes[pref]))) # replace
                for pref in sorted(cur.nodes.keys(), key=lambda x : self.err_model[last_symb+'_'][1][x])[:k]: # [:k]
                    self.queue.put((-self.loss_function(cur_lev + self.err_model[last_symb+'_'][1][pref] * 0.2, cur.nodes[pref].count * 2, self.lang_model[cur_pref+pref[-1]][0], a, b, c), 
                                    (cur_pos, cur_pref+pref[-1], cur_lev + self.err_model[last_symb+'_'][1][pref] * 0.2, cur.nodes[pref]))) # insert
                self.queue.put((-self.loss_function(cur_lev + self.err_model[last_symb+word[cur_pos]][1][last_symb+'_'] * 2, cur.count / 4, self.lang_model[cur_pref][0], a, b, c), 
                                (cur_pos+1, cur_pref, cur_lev + self.err_model[last_symb+word[cur_pos]][1][last_symb+'_'] * 2, cur))) # delete
            else:
                for pref in sorted(cur.nodes.keys(), key=lambda x : self.loss_function(cur_lev + self.err_model['^'+word[cur_pos]][1][x], cur.nodes[x].count, self.lang_model[cur_pref+x[-1]][0], a, b, c))[-k:]:
                    self.queue.put((-self.loss_function(cur_lev + self.err_model['^'+word[cur_pos]][1][pref], cur.nodes[pref].count, self.lang_model[cur_pref+pref[-1]][0], a, b, c), 
                                    (cur_pos+1, cur_pref+pref[-1], cur_lev + self.err_model['^'+word[cur_pos]][1][pref], cur.nodes[pref]))) # replace
                for pref in sorted(cur.nodes.keys(), key=lambda x : self.err_model['^_'][1][x])[:k]: # [:k]
                    self.queue.put((-self.loss_function(cur_lev + self.err_model['^_'][1][pref] * 0.2, cur.nodes[pref].count * 2, self.lang_model[cur_pref+pref[-1]][0], a, b, c), 
                                    (cur_pos, cur_pref+pref[-1], cur_lev + self.err_model['^_'][1][pref] * 0.2, cur.nodes[pref]))) # insert
                self.queue.put((-self.loss_function(cur_lev + self.err_model['^'+word[cur_pos]][1]['^_'] * 2, cur.count / 4, self.lang_model[cur_pref][0], a, b, c), 
                                (cur_pos+1, cur_pref, cur_lev + self.err_model['^'+word[cur_pos]][1]['^_'] * 2, cur))) # delete
        if cur_len != 0:
            return res
        else:
            return [(0, word)]
    
    def graph_prob(self, word, words):
        return max(map(lambda x : self.lang_model.query_prob(word+' '+x[1]), words))
    
    def recover_register(self, orig, fix):
        if len(orig) != len(fix):
            return fix
        res = ''
        for char_o, char_f in zip(orig, fix):
            res += char_f.upper() if char_o.isupper() else char_f
        return res

    def is_english(self, s):
        '''
        takes only lower words
        '''
        is_ok = re.search(r'\b[a-z0-9_]+\b', s, flags=re.IGNORECASE) is not None
        return is_ok

    def is_russian(self, s):
        '''
        takes only lower words
        '''
        is_ok = re.search(r'\b[а-я0-9_]+\b', s) is not None
        return is_ok
    
    def change_layout(self, query): # do smth with non russian/ukrainian words
        res = ''
        lang = None
        if self.is_english(query):
            for char in query:
                try:
                    res += self.layout[char]
                except KeyError:
                    res += char
            lang = 0 # for english
        elif self.is_russian(query):
            for char in query:
                try:
                    res += self.inv_layout[char]
                except KeyError:
                    res += char
            lang = 1 # for russian
        else:
            return query, False, lang
        return res, True, lang
    
    def loss_for_layout(self, word): # loss for generated not from search word
        length = len(word)
        if length == 0:
            return -1000
        try:
            cur = self.tree.head.nodes['^'+word[0]]
        except KeyError:
            return self.loss_function(0.7, 1, self.lang_model[word][0])# 0.7 cost of changing layout, may be different
        pos = 1
        while pos < length:
            try:
                cur = cur.nodes[word[pos-1:pos+1]]
            except KeyError:
                return self.loss_function(0.7, 1, self.lang_model[word][0])# 0.7 cost of changing layout, may be different
            pos += 1
        return self.loss_function(0, cur.count, self.lang_model[word][0])
    
    def loss_for_join(self, word):
        if word in self.lang_model.model_dict:
            return self.lang_model[word][0]*100000
        else:
            length = len(word)
            if length == 0:
                return -1000
            try:
                cur = self.tree.head.nodes['^'+word[0]]
            except KeyError:
                return self.loss_function(0.5, 1, self.lang_model[word][0])# 0.5 cost of joining, may be different
            pos = 1
            while pos < length:
                try:
                    cur = cur.nodes[word[pos-1:pos+1]]
                except KeyError:
                    return self.loss_function(0.5, 1, self.lang_model[word][0])# 0.5 cost of joining, may be different
                pos += 1
            return self.loss_function(0, cur.count * 4, self.lang_model[word][0])
    
    def extract_features(self, query):
        cur = []
        words = extract_words(query)
        cur.append(self.lang_model.query_prob(query)) # probability of this query in lang model
        cur.append(len(query)) # length of query
        cur.append(len(words)) # amount of words
        cur.append(len(re.findall(r'[^\w]', query)))# amount of non word symbols
        amount_in_voc = 0
        for word in words:# amount of words in vocabulary
            if self.lang_model[word][0] != self.lang_model.smooth_const:
                amount_in_voc += 1
        cur.append(amount_in_voc)
        return cur
    
    def query_decision(self, ):
        pass
    
    def cust_max(self, mas, key): # for query_prob
        cur_max = mas[0] if len(mas) > 0 else (-10000, '')
        for elem in mas[1:]:
            if key(elem) > key(cur_max):
                cur_max = elem
            elif key(elem) == key(cur_max):
                cur_split = re.findall(r'\w+', elem[1])
                cur_split_score = 1
                for word in cur_split:
                    cur_split_score *= self.lang_model[word][0]
                cur_max_split = re.findall(r'\w+', cur_max[1])
                cur_max_split_score = 1
                for word in cur_max_split:
                    cur_max_split_score *= self.lang_model[word][0]
                cur_max = elem if cur_split_score > cur_max_split_score else cur_max
        return cur_max
    
    def join(self, cur_word, word):
        joined = False
        new_word = cur_word+word
        if self.lang_model[new_word][0] > self.lang_model.smooth_const:
            joined = True
            new_word = (self.loss_for_join(new_word), new_word)
        return (new_word, joined)
    
    def split(self):
        pass
    
    def cust_erase(self, mas, word):
        elem_to_erase = None
        found = False
        for elem in mas:
            if word == elem[1]:
                elem_to_erase = elem
                found = True
                break
        if found:
            mas.remove(elem_to_erase)
    
    def recover_non_words(self, query, non_words, reg_words):
        words = extract_words(query)
        words_len = len(words)
        non_len = len(non_words)
        reg_len = len(reg_words)
        if reg_len != words_len:
            if words_len == non_len+1:
                fixed_query = words[0]
                for word, non_word in zip(words[1:], non_words):
                    fixed_query += non_word + word
            elif words_len == non_len:
                fixed_query = ''
                for word, non_word in zip(words, non_words):
                    fixed_query += word + non_word
            else:
                return query
        else:
            if words_len == non_len+1:
                fixed_query = self.recover_register(reg_words[0], words[0])
                for word, non_word, reg_word in zip(words[1:], non_words, reg_words[1:]):
                    fixed_query += non_word + self.recover_register(reg_word, word)
            elif words_len == non_len:
                fixed_query = ''
                for word, non_word, reg_word in zip(words, non_words, reg_words):
                    fixed_query += self.recover_register(reg_word, word) + non_word
            else:
                return query
        return fixed_query
    
    def search_query(self, query): # split/join/layout only on first iteration, + add iterations
        iteration = 0
        fixed_query = query
        reg_words = extract_words_register(query) # may need to place inside cycle
        non_words = extract_non_words(query) # may need to place inside cycle
        non_length = len(non_words)
        pos_recover = 0
        while iteration < 2: #amount of iterations
            iteration += 1
            query = fixed_query
            words = extract_words(query)
            words_length = len(words)
            if iteration == 1 and (len(words) > 8 or len(words) < 1): # too big or empty query
                return query
            changes = 0
            if self.boost.predict_proba(np.array(self.extract_features(query)).reshape(1, -1))[0][0] >= 0.995:# classifier decides does query need fix
                tmp_word, changed, lang_type = self.change_layout(query)
                if changed:
                    if tmp_word in self.rus_to_eng if lang_type == 0 else self.eng_to_rus: # customize max
                        return self.cust_max([(self.lang_model.query_prob(query), query), (self.lang_model.query_prob(tmp_word), 
                                                                                  tmp_word)], key=lambda x : x[0])[1]
                    else:
                        return self.recover_non_words(query, non_words, reg_words)
                else:
                    return self.recover_non_words(query, non_words, reg_words)
            query_res = []
            for word in words:
                if self.boost.predict_proba(np.array(self.extract_features(word)).reshape(1, -1))[0][0] >= 0.995: # classifier decides does query need fix
                    query_res.append([(self.lang_model[word][0]*100, word)]) # experimental
                else:
                    res = self.search(word)
                    query_res.append(res)
            for i, word in enumerate(words): # layouts
                tmp_word, changed, lang_type = self.change_layout(words[i])
                if changed: # layout
                    if tmp_word in self.rus_to_eng if lang_type == 0 else self.eng_to_rus:
                        query_res[i].append((self.loss_for_layout(tmp_word), tmp_word))
            query_len = len(query_res)
            if query_len == 1: # query of 1 word
                word_changed = max(query_res[0], key=lambda x : x[0])[1] 
                if word_changed == words[0]:
                    return self.recover_register(reg_words[0], word_changed)
                else:
                    changes += 1
                    fixed_query = word_changed
                    continue
            pos = 0
            next_pos = 1
            if iteration == 1:
                joins = []
                erase = False
                for cur_word in query_res[pos]:
                    if not erase: # delete
                        for word in query_res[next_pos]: # joins
                            pot_join, joined = self.join(cur_word[1], word[1])
                            if joined: # delete from next ?
                                joins.append(pot_join)
                                erase = True
                                break # delete
                if erase:
                    next_pos += 1
                    query_res[pos] = joins
            if query_len > next_pos: # if joined first two words then no way to join another one(may be in next iteration)
                cur_word = self.cust_max(query_res[pos], key=lambda x : self.graph_prob(x[1], query_res[next_pos]))# [1]
                if cur_word[1] != words[0]:
                    changes += 1
            else: # after joining len query_res == 1
                word_changed = max(query_res[pos], key=lambda x : x[0])# [1]
                if word_changed == words[0]:
                    return word_changed
                else:
                    changes += 1
                    fixed_query = word_changed[1]
                    continue
            pos = next_pos
            next_pos += 1
            fixed_query = cur_word[1]
            while pos < query_len: # query_len
                if next_pos != query_len:
                    if iteration == 1:
                        joins = []
                        erase = False
                        for new_word in query_res[pos]:
                            if not erase: #delete
                                for word in query_res[next_pos]:
                                    pot_join, joined = self.join(new_word[1], word[1])
                                    if joined:
                                        joins.append(pot_join)
                                        erase = True
                                        break # delete
                        if erase:
                            next_pos += 1
                            query_res[pos] = joins
                        cur_word = self.cust_max(query_res[pos], key=lambda x : self.lang_model.query_prob(cur_word[1]+' '+x[1]))
                    else:
                        cur_word = self.cust_max(query_res[pos], key=lambda x : self.lang_model.query_prob(cur_word[1]+' '+x[1]))
                else:
                    cur_word = self.cust_max(query_res[pos], key=lambda x : self.lang_model.query_prob(cur_word[1]+' '+x[1]))
                if cur_word[1] != words[pos]:
                        changes += 1
                fixed_query += ' ' + cur_word[1]
                pos = next_pos
                next_pos += 1
            if changes == 0:
                return self.recover_non_words(fixed_query, non_words, reg_words)
        return self.recover_non_words(fixed_query, non_words, reg_words)

layout = {'a' : 'ф','b' : 'и','c' : 'с','d' : 'в','e' : 'у','f' : 'а','g' : 'п','h' : 'р','i' : 'ш','j' : 'о','k' : 'л','l' : 'д','m' : 'ь','n' : 'т','o' : 'щ','p' : 'з','q' : 'й','r' : 'к','s' : 'ы','t' : 'е','u' : 'г','v' : 'м','w' : 'ц','x' : 'ч','y' : 'н','z' : 'я','`' : 'ё','[' : 'х',']' : 'ъ',';' : 'ж','\'' : 'э',',' : 'б','.' : 'ю','@' : '"','#' : '№','$' : ';','^' : ':','&' : '?','/' : '.','?' : ',','{' : 'Х','}' : 'Ъ','|' : '/','~' : 'Ё','\\' : '\\','1' : '1','2' : '2','3' : '3','4' : '4','5' : '5','6' : '6','7' : '7','8' : '8','9' : '9','0' : '0','(' : '(',')' : ')','*' : '*','%' : '%','!' : '!','-' : '-','=' : '=','_' : '_','+' : '+','A' : 'Ф','B' : 'И','C' : 'С','D' : 'В','E' : 'У','F' : 'А','G' : 'П','H' : 'Р','I' : 'Ш','J' : 'О','K' : 'Л','L' : 'Д','M' : 'Ь','N' : 'Т','O' : 'Щ','P' : 'З','Q' : 'Й','R' : 'К','S' : 'Ы','T' : 'Е','U' : 'Г','V' : 'М','W' : 'Ц','X' : 'Ч','Y' : 'Н','Z' : 'Я',':' : 'Ж','"' : 'Э','<' : 'Б','>' : 'Ю', ' ' : ' '}
inv_layout = {'ф' : 'a','и' : 'b','с' : 'c','в' : 'd','у' : 'e','а' : 'f','п' : 'g','р' : 'h','ш' : 'i','о' : 'j','л' : 'k','д' : 'l','ь' : 'm','т' : 'n','щ' : 'o','з' : 'p','й' : 'q','к' : 'r','ы' : 's','е' : 't','г' : 'u','м' : 'v','ц' : 'w','ч' : 'x','н' : 'y','я' : 'z','ё' : '`','х' : '[','ъ' : ']','ж' : ';','э' : '\'','б' : ',','ю' : '.','"' : '@','№' : '#',';' : '$',':' : '^','?' : '&','.' : '/',',' : '?','Х' : '{','Ъ' : '}','/' : '|','Ё' : '~','\\' : '\\','1' : '1','2' : '2','3' : '3','4' : '4','5' : '5','6' : '6','7' : '7','8' : '8','9' : '9','0' : '0','(' : '(',')' : ')','*' : '*','%' : '%','!' : '!','-' : '-','=' : '=','_' : '_','+' : '+','Ф' : 'A','И' : 'B','С' : 'C','В' : 'D','У' : 'E','А' : 'F','П' : 'G','Р' : 'H','Ш' : 'I','О' : 'J','Л' : 'K','Д' : 'L','Ь' : 'M','Т' : 'N','Щ' : 'O','З' : 'P','Й' : 'Q','К' : 'R','Ы' : 'S','Е' : 'T','Г' : 'U','М' : 'V','Ц' : 'W','Ч' : 'X','Н' : 'Y','Я' : 'Z','Ж' : ':','Э' : '"','Б' : '<','Ю' : '>', ' ' : ' '}

error_model = pick_undump('./err_model')
language_model = pick_undump('./lang_model')
# svd = pick_undump('./svd')
# tfidf = pick_undump('./tfidf')
classifier = cat.CatBoostClassifier()
classifier.load_model('./classifier1') # ./classifier
eng_to_rus = pick_undump('./eng_to_rus')
rus_to_eng = pick_undump('./rus_to_eng')
# bor = Bor(error_model, language_model, svd, tfidf, classifier, layout, inv_layout, eng_to_rus, rus_to_eng)
bor = Bor(error_model, language_model, classifier, layout, inv_layout, eng_to_rus, rus_to_eng)
bor.fit()
for query in sys.stdin :
    print(bor.search_query(re.sub(r'\n', '', query)))