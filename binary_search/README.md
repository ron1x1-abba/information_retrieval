## Realization of reverse index binary search

Here is reverse index binary search for web documents. Realization with fast intersections and unions through Tree, Simple9 compression/decompression of index and lexical analyzier for search queris.

Archive files `.gz` should contain urls and html docs in special format.

To run binary search : 
1) you should run `preinstall.sh` to install necessary lib's
2) run `make_index.sh` to create index for search
3) run `search.sh` to start searching for documents (directories or files with `.gz` archive(s) pass through arguments of a command line)

In [help](https://github.com/ron1x1-abba/information_retrieval/blob/main/binary_search/help.txt) is description in Russian of how code works.
