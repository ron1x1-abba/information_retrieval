## Realization of spellchecker through fuzzy search with bor.

Currently here is the realization of writing error correction with replacing wrong symbols, changing layuot and joining words.

In spellchecker I use language model and error model based on real search queries. There are also some ml models that help to improve qualitu of error correction.

To run code you must name your file with queries `queries_all.txt` , which contains `wrong<tab>right` queries. Then you should pretrain ml models in `Classifier.py`, run `indexer.py` and after that run `spellchecker.py` , which receives queries on stdin and returns corrected queries to stdout.
