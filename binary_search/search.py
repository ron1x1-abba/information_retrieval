import struct
import sys
import re
from collections import defaultdict

def decompress():
    '''
    Simple9 encoding. 4 bits -- selector; 28 bits -- code
    '''
    sel_mask = 0xF0000000 # first 4 bits are 1 another are 0
    code_mask = ~sel_mask
    masks = [0x08000000, 0x0C000000, 0x0E000000, 0x0F000000, 0x0F800000, 0x0FE00000, 0x0FF80000, 0x0FFFC000, 0x0FFFFFFF]
    index = defaultdict(list)
    with open('./index', mode='rb') as f, open('./terms', mode='r', encoding='utf-8') as fr :
        size = f.read(8)
        size_of_ind = struct.unpack('Q', size)[0]
        for i in range(size_of_ind) :
            term = fr.readline()[:-1]
            size = f.read(8)
            size_list = struct.unpack('Q', size)[0]
            tmp_list = []
            cur_ints = 0
            while cur_ints < size_list :
                code = f.read(4)
                code = struct.unpack('I', code)[0]
                selector = (code & sel_mask) >> 28
                nums = code & code_mask
                if selector == 0 : # 28 nums by 1 bit
                    cur_mask = masks[0]
                    for j in range(28) :
                        tmp_list.append((nums & cur_mask) >> 27)
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 1
                elif selector == 1 : # 14 nums by 2 bits
                    cur_mask = masks[1]
                    for j in range(14) :
                        tmp_list.append((nums & cur_mask) >> 26) # ??
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 2
                elif selector == 2 : # 9 nums by 3 bits
                    cur_mask = masks[2]
                    for j in range(9) :
                        tmp_list.append((nums & cur_mask) >> 25)
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 3
                elif selector == 3 : # 7 nums by 4 bits
                    cur_mask = masks[3]
                    for j in range(7) :
                        tmp_list.append((nums & cur_mask) >> 24)
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 4
                elif selector == 4 : # 5 nums by 5 bits
                    cur_mask = masks[4]
                    for j in range(5) :
                        tmp_list.append((nums & cur_mask) >> 23)
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 5
                elif selector == 5 : # 4 nums by 7 bits
                    cur_mask = masks[5]
                    for j in range(4) :
                        tmp_list.append((nums & cur_mask) >> 21)
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 7
                elif selector == 6 : # 3 nums by 9 bits
                    cur_mask = masks[6]
                    for j in range(3) :
                        tmp_list.append((nums & cur_mask) >> 19)
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 9
                elif selector == 7 : # 2 nums by 14 bits
                    cur_mask = masks[7]
                    for j in range(2) :
                        tmp_list.append((nums & cur_mask) >> 14)
                        cur_ints += 1
                        if cur_ints >= size_list :
                            break
                        nums <<= 14
                else : # 1 num by 28 bits
                    cur_mask = masks[8]
                    tmp_list.append(nums & cur_mask)
                    cur_ints += 1
            index[term] = list(reversed(tmp_list))
    return index

class Node:
    def __init__(self, term):
        self.l = None
        self.r = None
        self.term = term
        self.end = False
        if not ((term == '|') or (term == '&') or (term == '!')) :
            self.ids = rev_index[term.lower()]
            self.pos = 0
            self.op = False
            self.max = len(self.ids)
            self.end = True if self.max == 0 else False
        elif term == '!' :
            self.doc = -1
            self.op = True
            self.doc_all = 0
        else :
            self.doc = -1
            self.op = True

AND = 1
OR = 2
NOT = 3
L_PAR = 4
R_PAR = 5
TERM = 6
START = 7
OK = 8
END = 9
class lexer:
    def __init__(self, string):
        self.string = string
        self.pos = 0
        self.end = len(string)
        self.cur_type = None
        self.cur_text = ''
        self.c = self.string[self.pos]
        
    def l_next(self):
        self.cur_text = ''
        state = START
        while state != OK :
            if state == START :
                if self.c == '|' :
                    state = OR
                elif self.c == '&' :
                    state = AND
                elif self.c == '!' :
                    state = NOT
                elif self.c == '(' :
                    state = L_PAR
                elif self.c == ')' :
                    state = R_PAR
                elif self.c.isalnum() :
                    state = TERM
                    self.cur_text += self.c
                elif self.c.isspace() :
                    if self.pos < self.end :
                        self.pos += 1
                    else :
                        state = OK
                        self.cur_type = END
                elif self.c == '\n' :
                    state = OK
                    self.cur_type == END
            elif state == TERM :
                if self.c.isalnum() :
                    if self.pos < self.end :
                        self.cur_text += self.c
                    else :
                        self.cur_type = TERM
                        state = OK
                else :
                    self.cur_type = TERM
                    state = OK
            elif state == OR :
                self.cur_type = OR
                state = OK
            elif state == AND :
                self.cur_type = AND
                state = OK
            elif state == NOT :
                self.cur_type = NOT
                state = OK
            elif state == L_PAR :
                self.cur_type = L_PAR
                state = OK
            elif state == R_PAR :
                self.cur_type = R_PAR
                state = OK
            
            if state != OK:
                self.pos += 1
                if self.pos < self.end :
                    self.c = self.string[self.pos]
            if state == OK and self.pos > self.end:
                self.cur_type = END

'''
GRAMMARY : 
    E -> M { or M}
    M -> F { and F}
    R -> ( E ) | ! R | word
'''

class parser:

    def __init__(self, string):
        self.lexer = lexer(string)
        self.lexer.l_next()
    
    def E(self):
        left = self.M()
        while self.lexer.cur_type == OR :
            self.lexer.l_next()
            right = self.M()
            new = Node('|')
            new.l = left
            new.r = right
            go_tree(new) # experimental
            left = new
        return left
    
    def M(self):
        left = self.R()
        while self.lexer.cur_type == AND :
            self.lexer.l_next()
            right = self.R()
            new = Node('&')
            new.l = left
            new.r = right
            go_tree(new) # experimental
            left = new
        return left
    
    def R(self):
        left = None
        if self.lexer.cur_type == L_PAR :
            self.lexer.l_next()
            left = self.E()
            if self.lexer.cur_type != R_PAR :
                raise RuntimeError('Expected )')
            self.lexer.l_next()
        elif self.lexer.cur_type == TERM :
            left = Node(self.lexer.cur_text)
            self.lexer.l_next()
        elif self.lexer.cur_type == NOT :
            left = Node('!') # make smth
            self.lexer.l_next()
            left.l = self.R() # was self.E()
            go_tree(left) # experimental
        else :
            raise RuntimeError('Unexpected token. ( or TERM expected')
        return left

def go_tree(node):
    if node.r is None: # case that current node is NOT
        if not node.l.end :
            if node.l.op :
                node.doc = node.l.doc
                go_tree(node.l)
            else :
                node.doc = node.l.ids[node.l.pos]
                if node.l.pos < node.l.max - 1 :
                    node.l.pos += 1
                else :
                    node.l.end = True
        else :
            node.end = True
    elif (not node.l.op) and (not node.r.op) : # case that left and right are terms
        if node.term == '&' : # intersection
            if not (node.l.end or node.r.end) :
                l_pos, r_pos, found = fast_intersec(node.l, node.r) # do not forget to make addition of position
                if found :
                    node.doc = node.l.ids[l_pos]
                    node.l.pos = l_pos + 1
                    if node.l.pos == node.l.max:
                        node.l.end = True
                    node.r.pos = r_pos + 1
                    if node.r.pos == node.r.max:
                        node.r.end = True
                else :
                    node.l.pos = l_pos
                    node.l.end = True
                    node.r.pos = r_pos
                    node.r.end = True
                    node.end = True
            else :
                node.end = True
                node.l.end = True
                node.r.end = True
        elif node.term == '|' : # seems to be ready
            if not (node.l.end or node.r.end) :
                if node.l.ids[node.l.pos] < node.r.ids[node.r.pos] :
                    node.doc = node.l.ids[node.l.pos]
                    if node.l.pos < node.l.max - 1 :
                        node.l.pos += 1
                    else :
                        node.l.end = True
                elif node.l.ids[node.l.pos] > node.r.ids[node.r.pos] :
                    node.doc = node.r.ids[node.r.pos]
                    if node.r.pos < node.r.max - 1 :
                        node.r.pos += 1
                    else :
                        node.r.end = True
                else :
                    node.doc = node.l.ids[node.l.pos]
                    if node.l.pos < node.l.max - 1 :
                        node.l.pos += 1
                    else :
                        node.l.end = True
                    if node.r.pos < node.r.max - 1 :
                        node.r.pos += 1
                    else :
                        node.r.end = True
            elif node.l.end and node.r.end :
                node.end = True
            elif node.l.end : # left ended
                node.doc = node.r.ids[node.r.pos]
                if node.r.pos < node.r.max - 1 :
                    node.r.pos += 1
                else :
                    node.r.end = True
            else : # right ended
                node.doc = node.l.ids[node.l.pos]
                if node.l.pos < node.l.max - 1 :
                    node.l.pos += 1
                else :
                    node.l.end = True
        else : # not
            print('Impossible!')
            return
    elif (node.l.op and node.r.op): # case that both are operations
        if node.term == '&' : # intersection
            if not (node.l.term == '!' or node.r.term == '!'):
                if not (node.l.end or node.r.end) :
                    doc = fast_intersec_both(node.l, node.r)
                    if doc == -1 :
                        node.end = True
                        node.l.end = True
                        node.r.end = True
                    else :
                        node.doc = doc
                else :
                        node.l.end = True
                        node.r.end = True
                        node.end = True
            elif (node.r.term == '!' and node.l.term == '!') :
                if not (node.l.end or node.r.end) :
                    doc = fast_intersec_both_both_not(node.l, node.r)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        node.end = True
                        node.l.end = True
                        node.r.end = True
                else :
                    node.end = True
            elif node.l.term == '!' : # left is NOT operation
                if not (node.l.end or node.r.end) :
                    doc = fast_intersec_both_not(node.l, node.r)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        if node.r.end :
                            node.l.end = True
                            node.end = True
                        else :
                            node.doc = node.r.doc
                            go_tree(node.r)
                elif (node.l.end and node.r.end):
                    node.l.end = True
                    node.r.end = True
                    node.end = True
                elif node.l.end :
                    node.doc = node.r.doc
                    go_tree(node.r)
                else :
                    node.l.end = True
                    node.r.end = True
                    node.end = True
            else :
                if not (node.l.end or node.r.end) :
                    doc = fast_intersec_both_not(node.r, node.l)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        if node.l.end :
                            node.r.end = True
                            node.end = True
                        else :
                            node.doc = node.l.doc
                            go_tree(node.l)
                elif node.l.end and node.r.end :
                    node.r.end = True
                    node.l.end =True
                    node.end = True
                elif node.r.end :
                    node.doc = node.l.doc
                    go_tree(node.l)
                else :
                    node.r.end = True
                    node.l.end = True
                    node.end = True
        elif node.term == '|' : # addition # may be should check for empty list
            if not (node.l.term == '!' or node.r.term == '!'):
                if not (node.l.end or node.r.end) :
                    if node.l.doc < node.r.doc:
                        node.doc = node.l.doc
                        go_tree(node.l)
                    elif node.r.doc < node.l.doc:
                        node.doc = node.r.doc
                        go_tree(node.r)
                    else :
                        node.doc = node.l.doc
                        go_tree(node.l)
                        go_tree(node.r)
                elif node.l.end and node.r.end :
                    node.end = True
                elif node.l.end : # left ended
                    node.doc = node.r.doc
                    go_tree(node.r)
                else : # right ended
                    node.doc = node.l.doc
                    go_tree(node.l)
            elif (node.r.term == '!' and node.l.term == '!') :
                if not (node.l.end or node.r.end) :
                    doc = unific_both_both_not(node.l, node.r)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        if node.l.end :
                            node.r.end = True
                            node.end = True
                        elif node.r.end : # !!!!!!
                            node.l.end = True
                            node.end = True
                else :
                    node.l.end = True
                    node.r.end = True
                    node.end = True
            elif node.l.term == '!' : # left is NOT operation
                if not (node.l.end or node.r.end) :
                    doc = unific_both_not(node.l, node.r)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        if node.l.end :
                            node.r.end = True
                            node.end = True
                elif node.l.end and node.r.end :
                    node.end = True
                elif node.l.end :
                    node.r.end = True
                    node.end = True
                else : # right is ended
                    while not node.l.end :
                        if node.l.doc_all == node.l.doc :
                            node.l.doc_all += 1
                            if node.l.doc_all < amount_of_docs :
                                go_tree(node.l)
                                if node.l.end :
                                    node.l.end = False
                            else :
                                node.l.end = True
                        else :
                            node.doc = node.l.doc_all
                            node.l.doc_all += 1
                            if node.l.doc_all == amount_of_docs :
                                node.l.end = True
                            break
            else : # right is NOT operation
                if not (node.l.end or node.r.end) :
                    doc = unific_both_not(node.r, node.l)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        if node.r.end :
                            node.l.end = True
                            node.end = True
                elif node.l.end and node.r.end :
                    node.end = True
                elif node.r.end :
                    node.l.end = True
                    node.end = True
                else : # left is ended
                    while not node.r.end :
                        if node.r.doc_all == node.r.doc :
                            node.r.doc_all += 1
                            if node.r.doc_all < amount_of_docs :
                                go_tree(node.r)
                                if node.r.end :
                                    node.r.end = False
                            else :
                                node.r.end = True
                        else :
                            node.doc = node.r.doc_all
                            node.r.doc_all += 1
                            if node.r.doc_all == amount_of_docs :
                                node.r.end = True
                            break
        else : # not
            print('Impossible')
            return # finish last
    elif node.l.op : # case that right is term and left is operation
        if node.term == '&' : # intersection
            if not node.l.term == '!' :
                if not (node.l.end or node.r.end) :
                    r_pos, found = fast_intersec_node(node.l, node.r)
                    if found:
                        node.doc = node.l.doc
                        node.r.pos = r_pos + 1
                        if node.r.pos >= node.r.max :
                            node.r.end = True
                    else :
                        node.r.pos = r_pos
                        node.r.end = True
                        node.end = True
                else :
                        node.l.end = True
                        node.r.pos = node.r.max
                        node.r.end = True
                        node.end = True
            else :
                if not (node.l.end or node.r.end) :
                    r_pos, found = fast_intersec_node_not(node.l, node.r)
                    if found :
                        node.doc = node.r.ids[node.r.pos]
                        node.r.pos = r_pos + 1
                        if node.r.pos == node.r.max:
                            node.r.end = True
                        elif node.l.doc < node.r.ids[node.r.pos]:
                            go_tree(node.l)
                    else : # left is ended or right is ended
                        if node.l.end:
                            if r_pos < node.r.max:
                                node.doc = node.r.ids[node.r.pos]
                                node.r.pos = r_pos + 1
                            else :
                                node.r.end = True
                                node.end = True
                        else : # ?????
                            node.l.end = True
                            node.end = True
                elif node.l.end and node.r.end:
                    node.l.end = True
                    node.r.end = True
                    node.end = True
                elif node.l.end :
                    if node.r.pos < node.r.max:
                        node.doc = node.r.ids[node.r.pos]
                        node.r.pos = node.r.pos + 1
                    else :
                        node.r.end = True
                        node.end = True
                else :
                    node.l.end = True
                    node.end = True
        elif node.term == '|' : # addition # may be should check for empty list
            if not node.l.term == '!' :
                if not (node.l.end or node.r.end) :
                    if node.l.doc < node.r.ids[node.r.pos] :
                        node.doc = node.l.doc
                        go_tree(node.l)
                    elif node.r.ids[node.r.pos] < node.l.doc :
                        node.doc = node.r.ids[node.r.pos]
                        if node.r.pos < node.r.max -  1:
                            node.r.pos += 1
                        else :
                            node.r.end = True
                    else :
                        node.doc = node.l.doc
                        go_tree(node.l)
                        if node.r.pos < node.r.max -  1:
                            node.r.pos += 1
                        else :
                            node.r.end = True
                elif  node.l.end and node.r.end:
                    node.end = True
                elif node.l.end : # left ended
                    node.doc = node.r.ids[node.r.pos]
                    if node.r.pos < node.r.max - 1 :
                        node.r.pos += 1
                    else :
                        node.r.end = True
                else : # right ended
                    node.doc = node.l.doc
                    go_tree(node.l)
            else :
                if not (node.l.end or node.r.end) :
                    doc = unific_node_not(node.l, node.r)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        if node.l.end :
                            node.r.end = True
                            node.end = True
                        else :
                            while not node.l.end :
                                if node.l.doc_all == node.l.doc :
                                    node.l.doc_all += 1
                                    if node.l.doc_all < amount_of_docs :
                                        go_tree(node.l)
                                        if node.l.end :
                                            node.l.end = False
                                    else :
                                        node.l.end = True
                                else :
                                    node.doc = node.l.doc_all
                                    node.l.doc_all += 1
                                    if node.l.doc_all == amount_of_docs :
                                        node.l.end = True
                                    break
                elif  node.l.end and node.r.end:
                    node.end = True
                elif node.l.end : # left ended
                    node.r.end = True
                    node.end = True
                else :
                    while not node.l.end:
                        if node.l.doc_all == node.l.doc :
                            node.l.doc_all += 1
                            if node.l.doc_all < amount_of_docs :
                                go_tree(node.l)
                                if node.l.end :
                                    node.l.end = False
                            else :
                                node.l.end = True
                        elif node.l.doc_all < node.l.doc :
                            node.doc = node.l.doc_all
                            node.l.doc_all += 1
                            if node.l.doc_all == amount_of_docs :
                                node.l.end = True
                            break
                        else :
                            node.doc = node.l.doc_all 
                            node.l.doc_all += 1
                            if node.l.doc_all == amount_of_docs :
                                node.l.end = True
                            break
        else : # not
            return # finish last
    else : # case that left is term and right is operation # commutative with previous
        if node.term == '&' : # intersection
            if not node.r.term == '!' :
                if not (node.l.end or node.r.end) :
                    l_pos, found = fast_intersec_node(node.r, node.l)
                    if found:
                        node.doc = node.r.doc
                        node.l.pos = l_pos + 1
                        if node.l.pos >= node.l.max :
                            node.l.end = True
                    else :
                        node.l.pos = l_pos
                        node.l.end = True
                        node.end = True
                else :
                        node.l.pos = node.l.max
                        node.l.end = True
                        node.r.end = True
                        node.end = True
            else :
                if not (node.l.end or node.r.end) :
                    l_pos, found = fast_intersec_node_not(node.r, node.l)
                    if found :
                        node.doc = node.l.ids[node.l.pos]
                        node.l.pos = l_pos + 1
                        if node.l.pos == node.l.max:
                            node.l.end = True
                        elif node.r.doc < node.l.ids[node.l.pos]:
                            go_tree(node.r)
                    else : # left is ended or right is ended
                        if node.r.end:
                            if l_pos < node.l.max :
                                node.doc = node.l.ids[node.l.pos]
                                node.l.pos = l_pos + 1
                            else :
                                node.l.end = True
                                node.end = True
                        else : # ?????
                            node.r.end = True
                            node.end = True
                elif node.l.end and node.r.end :
                    node.r.end = True
                    node.end = True
                elif node.r.end :
                    if node.l.pos < node.l.max :
                        node.doc = node.l.ids[node.l.pos]
                        node.l.pos = node.l.pos + 1
                    else :
                        node.l.end = True
                        node.end = True
                else :
                    node.r.end = True
                    node.end = True
        elif node.term == '|' : # addition # may be should check for empty list
            if not node.r.term == '!' :
                if not (node.l.end or node.r.end) :
                    if node.r.doc < node.l.ids[node.l.pos] :
                        node.doc = node.r.doc
                        go_tree(node.r)
                    elif node.l.ids[node.l.pos] < node.r.doc:
                        node.doc = node.l.ids[node.l.pos]
                        if node.l.pos < node.l.max - 1 :
                            node.l.pos += 1
                        else :
                            node.l.end = True
                    else :
                        node.doc = node.r.doc
                        go_tree(node.r)
                        if node.l.pos < node.l.max - 1 :
                            node.l.pos += 1
                        else :
                            node.l.end = True
                elif node.l.end and node.r.end :
                    node.end = True
                elif node.r.end : # right ended
                    node.doc = node.l.ids[node.l.pos]
                    if node.l.pos < node.l.max - 1 :
                        node.l.pos += 1
                    else :
                        node.l.end = True
                else : # left ended
                    node.doc = node.r.doc
                    go_tree(node.r)
            else :
                if not (node.l.end or node.r.end) :
                    doc = unific_node_not(node.r, node.l)
                    if doc != -1 :
                        node.doc = doc
                    else :
                        if node.r.end :
                            node.l.end = True
                            node.end = True
                        else :
                            while not node.r.end :
                                if node.r.doc_all == node.r.doc :
                                    node.r.doc_all += 1
                                    if node.r.doc_all < amount_of_docs :
                                        go_tree(node.r)
                                        if node.r.end :
                                            node.r.end = False
                                    else :
                                        node.r.end = True
                                else :
                                    node.doc = node.r.doc_all
                                    node.r.doc_all += 1
                                    if node.r.doc_all == amount_of_docs :
                                        node.r.end = True
                                    break
                elif  node.l.end and node.r.end:
                    node.end = True
                elif node.r.end : # left ended
                    node.l.end = True
                    node.end = True
                else :
                    while not node.r.end:
                        if node.r.doc_all == node.r.doc :
                            node.r.doc_all += 1
                            if node.r.doc_all < amount_of_docs :
                                go_tree(node.r)
                                if node.r.end :
                                    node.r.end = False
                            else :
                                node.r.end = True
                        elif node.r.doc_all < node.r.doc :
                            node.doc = node.r.doc_all
                            node.r.doc_all += 1
                            if node.r.doc_all == amount_of_docs :
                                node.r.end = True
                            break
                        else :
                            node.doc = node.r.doc_all 
                            node.r.doc_all += 1
                            if node.r.doc_all == amount_of_docs :
                                node.r.end = True
                            break
        else : # not
            return # finish last

def LowerBound(A, key): 
    left = -1 
    right = len(A) 
    while right > left + 1: 
        middle = (left + right) // 2 
        if A[middle] >= key: 
            right = middle 
        else: 
            left = middle 
    return right

def fast_intersec(l, r):
    if l.max < r.max :
        for i in range(l.pos, l.max):
            right = LowerBound(r.ids[r.pos:], l.ids[i])
            if right < len(r.ids[r.pos:]) and r.ids[r.pos:][right] == l.ids[i]:
                return i, r.pos + right, True
            else : # change r.pos
                r.pos = r.pos + right
    else :
        for i in range(r.pos, r.max):
            right = LowerBound(l.ids[l.pos:], r.ids[i])
            if right < len(l.ids[l.pos:]) and l.ids[l.pos:][right] == r.ids[i]:
                return l.pos + right, i, True
            else : # change r.pos
                l.pos = l.pos + right
    return l.max, r.max, False

def fast_intersec_node(l, r) : # l is operation
    while not l.end :
        right = LowerBound(r.ids[r.pos:], l.doc)
        if right < len(r.ids[r.pos:]) and r.ids[r.pos:][right] == l.doc:
            return r.pos + right, True
        else : # change r.pos
            r.pos = r.pos + right
            go_tree(l)
    return r.max, False

def fast_intersec_node_not(l, r) : # l is NOT operation
    while not l.end :
        if r.ids[r.pos] == l.doc:
            go_tree(l)
            r.pos = r.pos + 1
            if r.pos == r.max:
                r.end = True
                return r.pos, False # means right ended
        elif r.ids[r.pos] > l.doc :
            go_tree(l)
        else :
            return r.pos, True
    return r.pos, False #means that left is ended

def fast_intersec_both(l, r) : # both is operations
    while not l.end:
        if l.doc < r.doc :
            go_tree(l)
        elif r.doc < l.doc :
            go_tree(r)
        else :
            cur_doc = l.doc
            go_tree(l)
            go_tree(r)
            return cur_doc
    return -1

def fast_intersec_both_not(l, r) : # left is NOT operation
    while not l.end :
        if l.doc < r.doc :
            go_tree(l)
        elif r.doc < l.doc :
            cur_doc = r.doc
            go_tree(r)
            return cur_doc
        else :
            go_tree(l)
            go_tree(r)
            if r.end :
                return -1
    return -1

def fast_intersec_both_both_not(l, r) : # both are NOT operation
    while (not l.end and not r.end) :
        if (l.doc_all != l.doc) and (r.doc_all != r.doc):
            cur_doc = l.doc_all
            l.doc_all += 1
            if l.doc_all == amount_of_docs :
                l.end = True
                r.end = True
            r.doc_all += 1
            return cur_doc
        elif (l.doc_all == l.doc) and (r.doc_all == r.doc) :
            l.doc_all += 1
            r.doc_all += 1
            if r.doc_all < amount_of_docs :
                go_tree(r)
                if r.end :
                    r.end = False
            else :
                l.end = True
                r.end = True
            if l.doc_all < amount_of_docs :
                go_tree(l)
                if l.end :
                    l.end = False
            else :
                l.end = True
                r.end = True
        elif l.doc_all == l.doc :
            l.doc_all += 1
            r.doc_all += 1
            if l.doc_all < amount_of_docs :
                go_tree(l)
                if l.end :
                    l.end = False
            else :
                l.end = True
                r.end = True
        else :
            l.doc_all += 1
            r.doc_all += 1
            if r.doc_all < amount_of_docs :
                go_tree(r)
                if r.end :
                    r.end = False
            else :
                l.end = True
                r.end = True
    return -1 # left and right is ended

def unific_node_not(l, r) : # left is NOT operation
    while not (l.end or r.end):
        if l.doc == l.doc_all :
            if r.ids[r.pos] == l.doc :
                cur_doc = r.ids[r.pos]
                r.pos += 1
                if r.pos == r.max :
                    r.end = True
                l.doc_all += 1
                if l.doc_all < amount_of_docs :
                    go_tree(l)
                    if l.end :
                        l.end = False
                else :
                    l.end = True
                return cur_doc
            if r.ids[r.pos] < l.doc :
                r.pos += 1
                if r.pos == r.max :
                    r.end = True
            else :
                l.doc_all += 1
                if l.doc_all < amount_of_docs :
                    go_tree(l)
                    if l.end :
                        l.end = False
                else :
                    l.end = True
        else :
            cur_doc = l.doc_all
            if l.doc_all == r.ids[r.pos] :
                r.pos += 1
                if r.pos == r.max :
                    r.end = True
            l.doc_all += 1
            if l.doc_all == amount_of_docs :
                l.end = True
            return cur_doc
    return -1

def unific_both_not(l, r) : # left is NOT operation, right is operation
    while not (l.end or r.end):
        if l.doc == l.doc_all :
            if r.doc == l.doc :
                cur_doc = r.doc
                go_tree(r)
                l.doc_all += 1
                if l.doc_all < amount_of_docs :
                    go_tree(l)
                    if l.end :
                        l.end = False
                else :
                    l.end = True
                return cur_doc
            else :
                l.doc_all += 1
                if l.doc_all < amount_of_docs :
                    go_tree(l)
                    if l.end :
                        l.end = False
                else :
                    l.end = True
        else :
            cur_doc = l.doc_all
            l.doc_all += 1
            if l.doc_all == amount_of_docs :
                l.end = True
            if l.doc_all > r.doc :
                go_tree(r)
            return cur_doc
    return -1 # case that l.end or r.end

def unific_both_both_not(l, r) : # both are NOT operations
    while not (l.end or r.end) :
        if l.doc == r.doc :
            if l.doc_all == l.doc :
                l.doc_all += 1
                if l.doc_all < amount_of_docs :
                    go_tree(l)
                    if l.end :
                        l.end = False
                else :
                    l.end = True
                r.doc_all += 1
                if r.doc_all < amount_of_docs :
                    go_tree(r)
                    if r.end :
                        r.end = False
                else :
                    r.end = True
            else :
                cur_doc = l.doc_all
                l.doc_all += 1
                if l.doc_all < amount_of_docs :
                    if l.doc < l.doc_all :
                        go_tree(l)
                        if l.end :
                            l.end = False
                else :
                    l.end = True
                r.doc_all += 1
                if r.doc_all < amount_of_docs :
                    if r.doc < r.doc_all :
                        go_tree(r)
                        if r.end :
                            r.end = False
                else :
                    r.end = True
                return cur_doc
        elif l.doc < r.doc :
            cur_doc = l.doc_all
            if l.doc_all == l.doc :
                l.doc_all += 1
                if l.doc_all < amount_of_docs :
                    go_tree(l)
                    if l.end :
                        l.end = False
                else :
                    l.end = True
                r.doc_all += 1
                if r.doc_all == amount_of_docs :
                    r.end = True
            else :
                l.doc_all += 1
                if l.doc_all == amount_of_docs :
                    l.end = True
                r.doc_all += 1
                if r.doc_all == amount_of_docs :
                    r.end = True
            return cur_doc
        else :
            cur_doc = r.doc_all
            if r.doc_all == r.doc :
                r.doc_all += 1
                if r.doc_all < amount_of_docs :
                    go_tree(r)
                    if r.end :
                        r.end = False
                else :
                    r.end = True
                l.doc_all += 1
                if l.doc_all == amount_of_docs :
                    l.end = True
            else :
                l.doc_all += 1
                if l.doc_all == amount_of_docs :
                    l.end = True
                r.doc_all += 1
                if r.doc_all == amount_of_docs :
                    r.end = True
            return cur_doc
    return -1 # case one of is ended

def go_tree_not(node) :
    res = []
    while node.doc_all < amount_of_docs :
        if node.doc_all == node.doc :
            node.doc_all += 1
            go_tree(node)
        else :
            res.append(node.doc_all)
            node.doc_all += 1
    return res

def flow_search(node): # at first node.doc must be equal -1
    if node.op :
        if node.term == '!':
            count = 1
            tmp_node = node
            while (tmp_node.l.term == '!') and (tmp_node.r is None) :
                tmp_node = tmp_node.l
                count += 1
            if count % 2 == 1 :
                return go_tree_not(node)
            else :
                return flow_search(tmp_node.l)
        else :
            if node.doc != -1:
                res = [node.doc]
            else :
                res = []
    else :
        return node.ids
    go_tree(node)
    while not node.end :
        res.append(node.doc)
        go_tree(node)
    return res

rev_index = decompress()
docs_url = []
with open('./urls', mode='r', encoding='utf-8') as f :
    for line in f :
        docs_url.append(line)
amount_of_docs = len(docs_url)
for query in sys.stdin :
    query = re.sub(r'\n', '', query)
    if len(query) == 0 :
        continue
    res = flow_search(parser(query).E())
    print(len(res))
    for i in res:
        print(docs_url[i], end='')
