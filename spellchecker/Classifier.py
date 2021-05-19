import numpy as np
from source import extract_words
from indexer import Language_model, undump_lang
import catboost as cat
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, accuracy_score
import pickle
import re

# need_fix = []
# no_need_fix = []

# with open('./queries_all.txt', encoding='utf-8') as f:
#     for line in f:
#         line = re.sub('\n', '', line.lower()).split('\t')
#         if len(line) == 2:
#             need_fix.append(line[0])
#             no_need_fix.append(line[1])
#         else:
#             no_need_fix.append(line[0])

# all_queries = need_fix + no_need_fix
# len_all_queries = len(all_queries)

# def cross_val(fix, no_fix, fix_targ, no_fix_targ, k=5):
#     """
#     Yields train and test arrays; train and test targets
#     """
#     amount = len_all_queries // k // 2 # 50 50 for fix and no fix
# #     amount = 9 // 2 // 2
#     tmp_fix = np.array(fix)
#     tmp_fix_targ = np.array(fix_targ)
#     tmp_no_fix = np.array(no_fix)
#     tmp_no_fix_targ = np.array(no_fix_targ)
#     no_fix_len = tmp_no_fix.shape[0]
#     while tmp_fix.shape[0] < no_fix_len:
#         tmp_fix = np.vstack((tmp_fix, fix))
#         tmp_fix_targ = np.vstack((tmp_fix_targ, fix_targ))
#     fix_perm = np.random.permutation(tmp_fix.shape[0])
#     no_fix_perm = np.random.permutation(tmp_no_fix.shape[0])
#     tmp_fix = tmp_fix[fix_perm]
#     tmp_no_fix = tmp_no_fix[no_fix_perm]
#     tmp_fix_targ = tmp_fix_targ[fix_perm]
#     tmp_no_fix_targ = tmp_no_fix_targ[no_fix_perm]
#     ind = np.random.permutation(no_fix_len)
#     for i in range(k):
#         cur_test_ind = ind[i*amount:(i+1)*amount]
#         cur_train_ind = np.hstack((ind[:i*amount], ind[(i+1)*amount:]))
#         yield (np.vstack((tmp_fix[cur_train_ind],tmp_no_fix[cur_train_ind])), np.vstack((tmp_fix[cur_test_ind],tmp_no_fix[cur_test_ind])), 
#                np.vstack((tmp_fix_targ[cur_train_ind],tmp_no_fix_targ[cur_train_ind])), np.vstack((tmp_fix_targ[cur_test_ind],tmp_no_fix_targ[cur_test_ind])))

# model = undump_lang('./lang_model')

# tfidf = TfidfVectorizer()
# tfidf.fit(all_queries)
# all_tfidf = tfidf.transform(all_queries)
# svd = TruncatedSVD(n_components=15)
# svd.fit(all_tfidf)

# def extract_features(query):
#     cur = []
#     words = extract_words(query)
#     cur.append(model.query_prob(query)) # probability of this query in lang model
#     cur.append(len(query)) # length of query
#     cur.append(len(words)) # amount of words
#     cur.append(len(re.findall(r'[^\w]', query)))# amount of non word symbols
#     amount_in_voc = 0
#     for word in words:# amount of words in vocabulary
#         if model[word][0] != model.smooth_const:
#             amount_in_voc += 1
#     cur.append(amount_in_voc)
#     return cur

# # creating features for classifier
# need_fix_mat = []
# no_need_fix_mat = []
# for query in need_fix:
#     cur = extract_features(query)
#     need_fix_mat.append(cur)
# need_fix_tfidf = tfidf.transform(need_fix)
# need_fix_mat = np.array(need_fix_mat)
# need_fix_mat = np.hstack((need_fix_mat, svd.transform(need_fix_tfidf)))
# for query in no_need_fix:
#     cur = extract_features(query)
#     no_need_fix_mat.append(cur)
# no_need_fix_tfidf = tfidf.transform(no_need_fix)
# no_need_fix_mat = np.array(no_need_fix_mat)
# no_need_fix_mat = np.hstack((no_need_fix_mat, svd.transform(no_need_fix_tfidf)))

# boost = cat.CatBoostClassifier(loss_function='Logloss', verbose=True, eval_metric='F1', iterations=150, thread_count=-1)
# boost.fit(np.vstack((need_fix_mat, no_need_fix_mat)), np.vstack((np.ones((need_fix_mat.shape[0], 1)), np.zeros((no_need_fix_mat.shape[0], 1)))), verbose=30)
# with open('./svd', mode='wb') as f:
#     pickle.dump(svd, f)
# with open('./tfidf', mode='wb') as f:
#     pickle.dump(tfidf, f)
# boost.save_model('./classifier')

def undump_svd(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)
    
def undump_tfidf(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)