import editdistance
import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict

@njit
def weighted_edit_distance(word1: str, word2: str, scores: np.ndarray, ct_labels_inv):
    m: int = len(word1)
    n: int = len(word2)
    dp = np.zeros((n+1, m+1), dtype=np.float32)
    dp[0, :] = np.arange(m+1)
    dp[:, 0] = np.arange(n+1)
    for i in range(1, n + 1):  ## word2
        for j in range(1, m + 1): ## word1
            delect_cost = _ed_delete_cost(j-1, i-1, word1, word2, scores, ct_labels_inv)  ## delect a[i]
            insert_cost = _ed_insert_cost(j-1, i-1, word1, word2, scores, ct_labels_inv)  ## insert b[j]
            if word1[j - 1] != word2[i - 1]:
                replace_cost = _ed_replace_cost(j-1, i-1, word1, word2, scores, ct_labels_inv) ## replace a[i] with b[j]
            else:
                replace_cost = 0
            dp[i][j] = min(dp[i-1][j] + insert_cost, dp[i][j-1] + delect_cost, dp[i-1][j-1] + replace_cost)

    return dp[n][m]

@njit
def _ed_delete_cost(j, i, word1, word2, scores, ct_labels_inv):
    ## delete a[i]
    return _get_score(scores[j], word1[j], ct_labels_inv)

@njit
def _ed_insert_cost(i, j, word1, word2, scores, ct_labels_inv):
    ## insert b[j]
    if i < len(word1) - 1:
        return (_get_score(scores[i], word1[i], ct_labels_inv) + _get_score(scores[i+1], word1[i+1], ct_labels_inv))/2
    else:
        return _get_score(scores[i], word1[i], ct_labels_inv)

@njit
def _ed_replace_cost(i, j, word1, word2, scores, ct_labels_inv):
    ## replace a[i] with b[j]
    # if word1 == "eeatpisaababarait".upper():
    #     print(scores[c2][i]/scores[c1][i])
    return max(1 - _get_score(scores[i], word2[j], ct_labels_inv)/_get_score(scores[i], word1[i], ct_labels_inv)*5, 0)

@njit
def _get_score(scores, char, ct_labels_inv):
    upper = ct_labels_inv[char.upper()]
    lower = ct_labels_inv[char.lower()]
    return max(scores[upper], scores[lower])

class LexiconMatcher:
    def __init__(self, dataset, lexicon_type, use_lexicon, ct_labels, weighted_ed=False):
        self.use_lexicon = use_lexicon
        self.lexicon_type = lexicon_type
        self.dataset = dataset
        self.ct_labels_inv = Dict.empty(
            key_type=types.string,
            value_type=types.int64,
        )
        for i, c in enumerate(ct_labels):
            self.ct_labels_inv[c] = i
        # maps char to index
        self.is_full_lex_dataset = "totaltext" in dataset or "ctw1500" in dataset or "inversetext" in dataset
        self._load_lexicon(dataset, lexicon_type)
        self.weighted_ed = weighted_ed

    def find_match_word(self, rec_str, img_id=None, scores=None):
        if not self.use_lexicon:
            return rec_str
        rec_str = rec_str.upper()
        dist_min = 100
        match_word = ''
        match_dist = 100

        lexicons = self.lexicons if self.lexicon_type != 3 else self.lexicons[img_id]
        pairs = self.pairs if self.lexicon_type != 3 else self.pairs[img_id]

        # scores of shape (seq_len, n_symbols) must be provided for weighted editdistance
        assert not self.weighted_ed or scores is not None

        for word in lexicons:
            word = word.upper()
            if self.weighted_ed:
                ed = weighted_edit_distance(rec_str, word, scores, self.ct_labels_inv)
            else:
                ed = editdistance.eval(rec_str, word)
            if ed < dist_min:
                dist_min = ed
                match_word = pairs[word]
                match_dist = ed
        
        if self.is_full_lex_dataset:
            # always return matched results for the full lexicon (for totaltext/ctw1500)
            return match_word
        else:
            # filter unmatched words for icdar
            return match_word if match_dist < 2.5 or self.lexicon_type == 1 else None

    @staticmethod
    def _get_lexicon_path(dataset):
        if "icdar2015" in dataset:
            g_lexicon_path = "datasets/evaluation/lexicons/ic15/GenericVocabulary_new.txt"
            g_pairlist_path = "datasets/evaluation/lexicons/ic15/GenericVocabulary_pair_list.txt"
            w_lexicon_path = "datasets/evaluation/lexicons/ic15/ch4_test_vocabulary_new.txt"
            w_pairlist_path = "datasets/evaluation/lexicons/ic15/ch4_test_vocabulary_pair_list.txt"
            s_lexicon_paths = [
                (str(fid+1), f"datasets/evaluation/lexicons/ic15/new_strong_lexicon/new_voc_img_{fid+1}.txt") for fid in range(500)]
            s_pairlist_paths = [
                (str(fid+1), f"datasets/evaluation/lexicons/ic15/new_strong_lexicon/pair_voc_img_{fid+1}.txt") for fid in range(500)]
        elif "totaltext" in dataset:
            s_lexicon_paths = s_pairlist_paths = None
            g_lexicon_path = "datasets/evaluation/lexicons/totaltext/tt_lexicon.txt"
            g_pairlist_path = "datasets/evaluation/lexicons/totaltext/tt_pair_list.txt"
            w_lexicon_path = "datasets/evaluation/lexicons/totaltext/weak_voc_new.txt"
            w_pairlist_path = "datasets/evaluation/lexicons/totaltext/weak_voc_pair_list.txt"
        elif "ctw1500" in dataset:
            s_lexicon_paths = s_pairlist_paths = w_lexicon_path = w_pairlist_path = None
            g_lexicon_path = "datasets/evaluation/lexicons/ctw1500/ctw1500_lexicon.txt"
            g_pairlist_path = "datasets/evaluation/lexicons/ctw1500/ctw1500_pair_list.txt"
        elif "inversetext" in dataset:
            s_lexicon_paths = s_pairlist_paths = w_lexicon_path = w_pairlist_path = None
            g_lexicon_path = "datasets/evaluation/lexicons/inversetext/inversetext_lexicon.txt"
            g_pairlist_path = "datasets/evaluation/lexicons/inversetext/inversetext_pair_list.txt"
        return g_lexicon_path, g_pairlist_path, w_lexicon_path, w_pairlist_path, s_lexicon_paths, s_pairlist_paths

    def _load_lexicon(self, dataset, lexicon_type):
        if not self.use_lexicon:
            return
        g_lexicon_path, g_pairlist_path, w_lexicon_path, w_pairlist_path, s_lexicon_path, s_pairlist_path = self._get_lexicon_path(
            dataset)
        if lexicon_type in (1, 2):
            # generic/weak lexicon
            lexicon_path = g_lexicon_path if lexicon_type == 1 else w_lexicon_path
            pairlist_path = g_pairlist_path if lexicon_type == 1 else w_pairlist_path
            if lexicon_path is None or pairlist_path is None:
                self.use_lexicon = False
                return
            with open(pairlist_path) as fp:
                pairs = dict()
                for line in fp.readlines():
                    line = line.strip()
                    if self.is_full_lex_dataset:
                        # might contain space in key word
                        split = line.split(' ')
                        half = len(split) // 2
                        word = ' '.join(split[:half]).upper()
                    else:
                        word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
            with open(lexicon_path) as fp:
                lexicons = []
                for line in fp.readlines():
                    lexicons.append(line.strip())
            self.lexicons = lexicons
            self.pairs = pairs
        elif lexicon_type == 3:
            # strong lexicon
            if s_lexicon_path is None or s_pairlist_path is None:
                self.use_lexicon = False
                return
            lexicons, pairlists = dict(), dict()
            for (fid, lexicon_path), (_, pairlist_path) in zip(s_lexicon_path, s_pairlist_path):
                with open(lexicon_path) as fp:
                    lexicon = []
                    for line in fp.readlines():
                        lexicon.append(line.strip())
                with open(pairlist_path) as fp:
                    pairs = dict()
                    for line in fp.readlines():
                        line = line.strip()
                        word = line.split(' ')[0].upper()
                        word_gt = line[len(word)+1:]
                        pairs[word] = word_gt
                lexicons[fid] = lexicon
                pairlists[fid] = pairs
            self.lexicons = lexicons
            self.pairs = pairlists
