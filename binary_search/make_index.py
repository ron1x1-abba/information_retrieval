import gzip
import codecs
import struct
import re
import sys
from collections import defaultdict
from os import listdir

def extract_url(doc) :
    return re.search(r'http:.*\b', doc, flags=re.DOTALL)

def extract_text(doc) :
    new_doc = re.sub(r'\xa0', ' ', doc, flags=re.DOTALL)
    new_doc = new_doc[1:]
    new_doc = re.sub(r'\n', ' ', new_doc, flags=re.DOTALL)
    return new_doc

def read_all_docs(paths):
    docs = []
    for path in paths:
        with gzip.open(path, 'rb') as file:
            while True:
                size = file.read(4)
                if size == b'':
                    break
                size = struct.unpack('i', size)[0]
                doc = file.read(size).decode('utf-8', 'ignore')
                doc = doc.split('\x1a')
                for j, i in enumerate(doc):
                    if len(i) < 2:
                        doc.pop(j)
                if len(doc) == 1:
                    doc.append('')
                url = extract_url(doc[0])
                url = url[0] if url is not None else None
                text = extract_text(doc[1])
                docs.append([url, text])
    return docs

def build_index(docs) :
    term_to_id = defaultdict(list)
    for num, doc in enumerate(docs):
        lowed = doc[1].lower()
        for term in set(re.findall(r'\w+', lowed)):
            if term in term_to_id:
                term_to_id[term].append(num)
            else :
                term_to_id[term] = []
                term_to_id[term].append(num)
    return term_to_id

paths = []
for arg in sys.argv[1:]:
    print(arg)
    if arg.endswith('gz') :
        paths.append(arg)
    else :
        paths = paths + ['./' + arg + '/' + i for i in listdir('./' + arg)]
# paths = sorted(paths)
docs = read_all_docs(paths)
docs_url = [i[0] + '/' for i in docs]
rev_index = build_index(docs)

def compress(dictionary):
    '''
    Simple9 encoding. 4 bits -- selector; 28 bits -- code
    '''
    with open('./index', mode='wb') as f, open('./terms',  mode='w', encoding='utf-8') as fw :
        f.write(struct.pack('Q', len(dictionary)))
        for term, mas in dictionary.items() :
            fw.write(term + '\n') # write term of dict
            f.write(struct.pack('Q', len(mas)))
            pos = len(mas) - 1
            while pos >= 0 :
                num = mas[pos]
                if num < 2 : # 2^1 and 28 chunks
                    res = 0
                    j = 27
                    for i in range(28) :
                        res |= num << j
                        pos -= 1
                        j -= 1
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                elif num < 4 :  # 2^2 and 14 chunks
                    res = 0x10000000
                    j = 26
                    for i in range(14) :
                        res |= num << j
                        pos -= 1
                        j -= 2
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                elif num < 8 : # 2^3 and 9 chunks, 1 bit lost
                    res = 0x20000000
                    j = 25
                    for i in range(9) :
                        res |= num << j
                        pos -= 1
                        j -= 3
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                elif num < 16 : # 2^4 and 7 chunks
                    res = 0x30000000
                    j = 24
                    for i in range(7) :
                        res |= num << j
                        pos -= 1
                        j -= 4
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                elif num < 32 : # 2^5 and 5 chunks, 5 bits lost
                    res = 0x40000000
                    j = 23
                    for i in range(5) :
                        res |= num << j
                        pos -= 1
                        j -= 5
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                elif num < 128 : # 2^7 and 4 chunks
                    res = 0x50000000
                    j = 21
                    for i in range(4) :
                        res |= num << j
                        pos -= 1
                        j -= 7
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                elif num < 512 : # 2^9 and 3 chunks, 1 bit lost
                    res = 0x60000000
                    j = 19
                    for i in range(3) :
                        res |= num << j
                        pos -= 1
                        j -= 9
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                elif num < 16384 : # 2^14 and 2 chunks
                    res = 0x70000000
                    j = 14
                    for i in range(2) :
                        res |= num << j
                        pos -= 1
                        j -= 14
                        if pos < 0 :
                            break
                        num = mas[pos]
                    f.write(struct.pack('I', res))
                else :
                    res = 0x80000000
                    res |= num
                    pos -= 1
                    if pos < 0 :
                        break
                    f.write(struct.pack('I', res))

compress(rev_index)

with open('./urls',  mode='w', encoding='utf-8') as fu :
    for url in docs_url :
        fu.write(url + '\n')
