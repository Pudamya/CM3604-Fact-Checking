# factcheck.py

import torch
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import spacy
import gc
import html
import math
import re
import unicodedata
from collections import Counter
from nltk.stem import PorterStemmer, SnowballStemmer


_NLTK_ENGLISH_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you youre youve youll youd your yours yourself yourselves he him his himself
she shes her hers herself it its itself they them their theirs themselves what which who whom this that thatll these
those am is are was were be been being have has had having do does did doing a an the and but if or because as until
while of at by for with about against between into through during before after above below to from up down in out on
off over under again further then once here there when where why how all any both each few more most other some such
no nor not only own same so than too very s t can will just don dont should shouldve now d ll m o re ve y ain aren
arent couldn couldnt didn didnt doesn doesnt hadn hadnt hasn hasnt haven havent isn isnt ma mightn mightnt mustn
mustnt needn neednt shan shant shouldn shouldnt wasn wasnt weren werent won wont wouldn wouldnt
""".split())

# Negations flip the meaning of a claim, so they are kept as content words rather than discarded.
NEGATION_WORDS = frozenset("""
no nor not never don dont ain aren arent couldn couldnt didn didnt doesn doesnt hadn hadnt hasn hasnt haven havent isn
isnt mightn mightnt mustn mustnt needn neednt shan shant shouldn shouldnt wasn wasnt weren werent won wont wouldn
wouldnt
""".split())

STOPWORDS = _NLTK_ENGLISH_STOPWORDS - NEGATION_WORDS

# Compiled once at import time.
_SENT_TAG_RE = re.compile(r"</?s>")                        # <s> ... </s> markers in the retrieved passages
_POSSESSIVE_RE = re.compile(r"['’]s\b")               # McAndrew's -> McAndrew
_APOSTROPHE_RE = re.compile(r"['’]")                  # O'Brien -> OBrien, don't -> dont
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")         # 1,000,000 -> 1000000
_TOKEN_RE_LOWER = re.compile(r"[a-z0-9]+")
_TOKEN_RE_CASED = re.compile(r"[A-Za-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def clean_text(text: Optional[str]) -> str:
    """Remove passage tags, accents and possessives; normalise apostrophes and digit groups."""
    if not text:
        return ""
    text = _SENT_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _POSSESSIVE_RE.sub("", text)
    text = _APOSTROPHE_RE.sub("", text)
    text = _THOUSANDS_RE.sub("", text)
    return text


def tokenize(text: Optional[str], lowercase: bool = True) -> List[str]:
    """Return ordered alphanumeric tokens after text cleaning."""
    text = clean_text(text)
    if lowercase:
        return _TOKEN_RE_LOWER.findall(text.lower())
    return _TOKEN_RE_CASED.findall(text)


def split_sentences(passage_text: Optional[str]) -> List[str]:
    """Split passage tags and sentence-final punctuation into non-empty sentences."""
    if not passage_text:
        return []
    sentences = []
    for chunk in _SENT_TAG_RE.split(passage_text):
        chunk = chunk.strip()
        if chunk:
            sentences.extend(s.strip() for s in _SENT_SPLIT_RE.split(chunk) if s.strip())
    return sentences


NORMALIZERS = ("none", "porter", "snowball", "lemma")


# --- PART 1: PREPROCESSING ---
class TextPreprocessor(object):
    """Clean, tokenise, filter and normalise text; cache repeated inputs."""

    def __init__(self, lowercase: bool = True, remove_stopwords: bool = True, normalizer: str = "porter",
                 min_token_len: int = 2):
        """Initialise configuration and validate supplied options."""
        if normalizer not in NORMALIZERS:
            raise ValueError("normalizer must be one of " + repr(NORMALIZERS))
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.normalizer = normalizer
        self.min_token_len = min_token_len
        self._stem_cache: Dict[str, str] = {}
        self._cache: Dict[Tuple[str, bool], Tuple[str, ...]] = {}
        self._stemmer = None
        self._nlp = None
        if normalizer == "porter":
            self._stemmer = PorterStemmer()
        elif normalizer == "snowball":
            self._stemmer = SnowballStemmer("english")
        elif normalizer == "lemma":
            # The lemmatiser only needs the tagger and attribute ruler; the parser and NER are disabled for speed.
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def config(self) -> dict:
        return {"lowercase": self.lowercase, "remove_stopwords": self.remove_stopwords,
                "normalizer": self.normalizer, "min_token_len": self.min_token_len}

    def _keep(self, token: str, filter_stopwords: bool) -> bool:
        if len(token) < self.min_token_len and not token.isdigit():
            return False
        return not (filter_stopwords and token.lower() in STOPWORDS)

    def _stem(self, token: str) -> str:
        stem = self._stem_cache.get(token)
        if stem is None:
            # NLTK stemmers lowercase their output; keep the case if lowercasing was switched off.
            stem = self._stemmer.stem(token, to_lowercase=False) if self.normalizer == "porter" \
                else self._stemmer.stem(token)
            self._stem_cache[token] = stem
        return stem

    def _run(self, text: str, filter_stopwords: bool) -> List[str]:
        if self._nlp is not None:
            tokens = []
            for tok in self._nlp(clean_text(text)):
                if filter_stopwords and tok.text.lower() in STOPWORDS:
                    continue
                for piece in tokenize(tok.lemma_, self.lowercase):
                    if self._keep(piece, filter_stopwords):
                        tokens.append(piece)
            return tokens
        tokens = [t for t in tokenize(text, self.lowercase) if self._keep(t, filter_stopwords)]
        if self._stemmer is not None:
            tokens = [self._stem(t) for t in tokens]
        return tokens

    def process(self, text: Optional[str]) -> List[str]:
        """Return normalised tokens, retaining duplicates for term frequencies."""
        text = text or ""
        key = (text, self.remove_stopwords)
        cached = self._cache.get(key)
        if cached is None:
            tokens = self._run(text, self.remove_stopwords)
            if not tokens and self.remove_stopwords:
                tokens = self._run(text, False)
            cached = tuple(tokens)
            self._cache[key] = cached
        return list(cached)

    def process_set(self, text: Optional[str]) -> frozenset:
        return frozenset(self.process(text))


def overlap_recall(fact: Iterable[str], passage: Iterable[str]) -> float:
    """ |F ∩ P| / |F| : the fraction of the fact's words that the passage contains. """
    fact, passage = set(fact), set(passage)
    return len(fact & passage) / len(fact) if fact else 0.0


def overlap_precision(fact: Iterable[str], passage: Iterable[str]) -> float:
    """ |F ∩ P| / |P| : the fraction of the passage's words that appear in the fact. """
    fact, passage = set(fact), set(passage)
    return len(fact & passage) / len(passage) if passage else 0.0


def overlap_f1(fact: Iterable[str], passage: Iterable[str]) -> float:
    """ Harmonic mean of overlap precision and recall. """
    p, r = overlap_precision(fact, passage), overlap_recall(fact, passage)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """ |A ∩ B| / |A ∪ B| (symmetric). """
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compute_idf(docs: Sequence[Iterable[str]]) -> Tuple[Dict[str, float], float]:
    """Smoothed IDF: log((1 + N) / (1 + document frequency)) + 1."""
    n_docs = len(docs)
    df = Counter(w for doc in docs for w in set(doc))
    idf = {w: math.log((1 + n_docs) / (1 + d)) + 1 for w, d in df.items()}
    return idf, math.log(1 + n_docs) + 1


def tfidf_vector(tokens: Iterable[str], idf: Dict[str, float], unseen_idf: float) -> Dict[str, float]:
    """ Sparse TF-IDF vector {word: count(word) * idf(word)}, with raw counts as term frequency. """
    return {w: c * idf.get(w, unseen_idf) for w, c in Counter(tokens).items()}


def cosine(u: Dict[str, float], v: Dict[str, float]) -> float:
    """ Cosine similarity of two sparse vectors; 0.0 if either is the zero vector. """
    if len(u) > len(v):
        u, v = v, u
    dot = math.fsum(x * v.get(w, 0.0) for w, x in u.items())
    norm_u = math.sqrt(math.fsum(x * x for x in u.values()))
    norm_v = math.sqrt(math.fsum(x * x for x in v.values()))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)


def tfidf_cosine(fact: Iterable[str], passage: Iterable[str], idf: Dict[str, float], unseen_idf: float) -> float:
    return cosine(tfidf_vector(fact, idf, unseen_idf), tfidf_vector(passage, idf, unseen_idf))


def idf_weighted_recall(fact: Iterable[str], passage: Iterable[str], idf: Dict[str, float],
                        unseen_idf: float) -> float:
    """Return fact-token recall weighted by smoothed IDF."""
    fact, passage = set(fact), set(passage)
    total = math.fsum(idf.get(w, unseen_idf) for w in fact)
    if total == 0.0:
        return 0.0
    return math.fsum(idf.get(w, unseen_idf) for w in fact & passage) / total


def fuzzy_recall(fact: Iterable[str], passage: Iterable[str], prefix: int = 5) -> float:
    """Recall with exact matches or shared prefixes of the requested length."""
    fact, passage = set(fact), set(passage)
    if not fact:
        return 0.0
    prefixes = {w[:prefix] for w in passage if len(w) >= prefix}
    found = sum(1 for w in fact if w in passage or (len(w) >= prefix and w[:prefix] in prefixes))
    return found / len(fact)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_recall(fact: Sequence[str], passage: Sequence[str], n: int = 2) -> float:
    """Clipped n-gram recall with the fact as reference."""
    fact, passage = list(fact), list(passage)
    reference = _ngrams(fact, n)
    total = sum(reference.values())
    if not total:
        return 0.0
    candidate = _ngrams(passage, n)
    return sum(min(count, candidate[gram]) for gram, count in reference.items()) / total


def rouge_l_recall(fact: Sequence[str], passage: Sequence[str]) -> float:
    """Longest common subsequence divided by the fact token count."""
    fact, passage = list(fact), list(passage)
    if not fact or not passage:
        return 0.0
    previous = [0] * (len(fact) + 1)
    for word in passage:
        current = [0] * (len(fact) + 1)
        for j, fact_word in enumerate(fact, 1):
            current[j] = previous[j - 1] + 1 if word == fact_word else max(previous[j], current[j - 1])
        previous = current
    return previous[-1] / len(fact)


def bleu(fact: Sequence[str], passage: Sequence[str], max_n: int = 4, epsilon: float = 0.1) -> float:
    """Sentence BLEU-4 with the fact as candidate and the passage as reference."""
    fact, passage = list(fact), list(passage)
    if not fact:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        candidate, reference = _ngrams(fact, n), _ngrams(passage, n)
        matches = sum(min(count, reference[gram]) for gram, count in candidate.items())
        denominator = max(1, sum(candidate.values()))
        if n == 1 and matches == 0:
            return 0.0
        log_sum += math.log((matches if matches else epsilon) / denominator) / max_n
    c, r = len(fact), len(passage)
    brevity = 1.0 if c > r else math.exp(1 - r / c)
    return brevity * math.exp(log_sum)


SET_METRICS = {"recall": overlap_recall, "precision": overlap_precision, "f1": overlap_f1, "jaccard": jaccard}
IDF_METRICS = {"tfidf_cosine": tfidf_cosine, "idf_recall": idf_weighted_recall}
METRICS = tuple(SET_METRICS) + tuple(IDF_METRICS)
EXTRA_SET_METRICS = {"fuzzy_recall": fuzzy_recall}
SEQUENCE_METRICS = {"rouge2": rouge_n_recall, "rougeL": rouge_l_recall, "bleu": bleu}
ALL_METRICS = METRICS + tuple(EXTRA_SET_METRICS) + tuple(SEQUENCE_METRICS)
AGGREGATIONS = ("max", "concat", "sentence_max")


def word_overlap_score(fact: str, passages: List[dict], preprocessor: TextPreprocessor, metric: str = "recall",
                       aggregation: str = "max", drop_title: bool = False) -> float:
    """Aggregate fact-to-passage lexical similarity using the selected metric."""
    if metric not in ALL_METRICS:
        raise ValueError("metric must be one of " + repr(ALL_METRICS))
    if aggregation not in AGGREGATIONS:
        raise ValueError("aggregation must be one of " + repr(AGGREGATIONS))
    if not passages:
        return 0.0

    fact_tokens = preprocessor.process(fact)
    texts = [p.get("text") or "" for p in passages]
    sentence_tokens = [preprocessor.process(s) for t in texts for s in split_sentences(t)]
    if aggregation == "max":
        units = [preprocessor.process(t) for t in texts]
    elif aggregation == "concat":
        units = [[w for t in texts for w in preprocessor.process(t)]]
    else:
        units = sentence_tokens

    if drop_title:
        title_set = set()
        for p in passages:
            title_set |= preprocessor.process_set(p.get("title"))
        kept = [w for w in fact_tokens if w not in title_set]
        if kept:  # a fact consisting only of the name keeps its tokens rather than becoming empty
            fact_tokens = kept
            units = [[w for w in unit if w not in title_set] for unit in units]
            sentence_tokens = [[w for w in s if w not in title_set] for s in sentence_tokens]

    if not fact_tokens:
        return 0.0
    if metric in SET_METRICS or metric in EXTRA_SET_METRICS or metric in SEQUENCE_METRICS:
        fn = SET_METRICS.get(metric) or EXTRA_SET_METRICS.get(metric) or SEQUENCE_METRICS[metric]
        return max((fn(fact_tokens, unit) for unit in units), default=0.0)
    idf, unseen_idf = compute_idf(sentence_tokens)
    fn = IDF_METRICS[metric]
    return max((fn(fact_tokens, unit, idf, unseen_idf) for unit in units), default=0.0)


class FactExample(object):
    """A fact, its retrieved passages and its gold label."""
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


# --- PART 2: NLI INFERENCE ---
class EntailmentModel(object):
    """Batched NLI inference with bounded input windows and automatic label mapping."""
    BATCH_SIZE = 16
    MAX_BATCH_TOKENS = 8192
    WINDOW_OVERLAP = 32

    def __init__(self, model, tokenizer, cuda=False, batch_size: Optional[int] = None,
                 max_batch_tokens: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.model.eval()
        self.batch_size = int(batch_size or self.BATCH_SIZE)
        self.max_batch_tokens = int(max_batch_tokens or self.MAX_BATCH_TOKENS)
        if self.batch_size < 1 or self.max_batch_tokens < 1:
            raise ValueError("batch_size and max_batch_tokens must be positive")
        self.entailment_index = self._resolve_label_index("entailment")
        self.contradiction_index = self._resolve_label_index("contradiction")
        limits = [512]  # maximum sequence length of the coursework DeBERTa checkpoint
        for limit in (getattr(tokenizer, "model_max_length", None),
                      getattr(model.config, "max_position_embeddings", None)):
            if isinstance(limit, int) and 0 < limit < 1_000_000:
                limits.append(limit)
        self.max_length = min(limits)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.premises_scored = 0
        self.pairs_scored = 0
        self.forward_passes = 0

    def _resolve_label_index(self, name: str) -> int:
        """Resolve the class index from model.config.id2label."""
        labels = getattr(self.model.config, "id2label", {}) or {}
        if len(labels) != 3:
            raise ValueError("Expected a three-class NLI model.")
        for index, label in labels.items():
            if str(label).strip().lower() == name:
                return int(index)
        checkpoint = str(getattr(self.model.config, "_name_or_path", "")).rstrip("/")
        generic = all(str(v).upper() == "LABEL_" + str(k) for k, v in labels.items())
        if generic and checkpoint == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli":
            return {"entailment": 0, "neutral": 1, "contradiction": 2}[name]
        raise ValueError("Cannot identify '" + name + "' in model.config.id2label.")

    def _windows(self, premise_ids: List[int], hypothesis_ids: List[int]) -> List[List[int]]:
        """ Splits the premise token ids into overlapping windows that fit next to the hypothesis. """
        budget = self.max_length - len(hypothesis_ids) - self.tokenizer.num_special_tokens_to_add(pair=True)
        if len(premise_ids) <= budget:
            return [premise_ids]
        overlap = min(self.WINDOW_OVERLAP, budget // 4)
        step = budget - overlap
        windows = []
        for start in range(0, len(premise_ids), step):
            windows.append(premise_ids[start:start + budget])
            if start + budget >= len(premise_ids):
                break
        return windows

    def _batches(self, features: List[Tuple[int, List[int], List[int]]]):
        """ Groups (owner, input_ids, token_type_ids) features, longest first, under both batch limits. """
        order = sorted(range(len(features)), key=lambda i: -len(features[i][1]))
        batch = []
        for i in order:
            longest = len(features[batch[0]][1]) if batch else len(features[i][1])
            if batch and (len(batch) >= self.batch_size or longest * (len(batch) + 1) > self.max_batch_tokens):
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def predict_probs(self, premises: Sequence[str], hypothesis: str) -> List[List[float]]:
        """
        Three-class probabilities for every (premise, hypothesis) pair.
        :param premises: evidence texts (premises); empty or whitespace-only premises get [0, 1, 0] (neutral)
        :param hypothesis: the fact
        :return: one [p_entailment, p_neutral, p_contradiction] list per premise, in the input order (the class order
        is normalised, whatever the model's own label order is). For a windowed premise, the window with the highest
        entailment probability is returned.
        """
        neutral = [0.0, 1.0, 0.0]
        results: List[Optional[List[float]]] = [None] * len(premises)
        hypothesis_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False, verbose=False)
        # A hypothesis can never use more than half of the input; facts are short, so this never triggers in practice
        # but it guarantees the code cannot crash on an unusual input.
        hypothesis_ids = hypothesis_ids[:self.max_length // 2]
        if not hypothesis_ids:
            return [list(neutral) for _ in premises]

        features = []
        # One batched tokenizer call for all premises (the fast tokenizer parallelises it). verbose=False: premises
        # longer than 512 tokens are expected here, they are windowed below.
        all_ids = self.tokenizer([p or "" for p in premises], add_special_tokens=False, verbose=False)["input_ids"] \
            if premises else []
        self.premises_scored += len(premises)
        for owner, premise_ids in enumerate(all_ids):
            if not premise_ids:
                results[owner] = list(neutral)
                continue
            for window in self._windows(premise_ids, hypothesis_ids):
                encoded = self.tokenizer.prepare_for_model(window, pair_ids=hypothesis_ids, add_special_tokens=True,
                                                           truncation=False, return_token_type_ids=True,
                                                           return_attention_mask=False, verbose=False)
                features.append((owner, encoded["input_ids"], encoded["token_type_ids"]))

        device = next(self.model.parameters()).device
        order = (self.entailment_index, 3 - self.entailment_index - self.contradiction_index, self.contradiction_index)
        with torch.inference_mode():
            for batch in self._batches(features):
                width = max(len(features[i][1]) for i in batch)
                ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
                types = torch.zeros((len(batch), width), dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):
                    length = len(features[i][1])
                    ids[row, :length] = torch.tensor(features[i][1], dtype=torch.long)
                    types[row, :length] = torch.tensor(features[i][2], dtype=torch.long)
                    mask[row, :length] = 1
                inputs = {"input_ids": ids.to(device), "token_type_ids": types.to(device),
                          "attention_mask": mask.to(device)}
                outputs = self.model(**inputs)
                logits = outputs.logits
                if tuple(logits.shape) != (len(batch), 3):
                    raise ValueError("Expected logits of shape (batch, 3), got " + str(tuple(logits.shape)))
                probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                for row, i in enumerate(batch):
                    p = [probabilities[row][k] for k in order]
                    if not all(math.isfinite(x) for x in p):
                        raise ValueError("Model produced a non-finite probability.")
                    owner = features[i][0]
                    if results[owner] is None or p[0] > results[owner][0]:
                        results[owner] = p
                self.forward_passes += 1
                self.pairs_scored += len(batch)
                # To prevent out-of-memory issues, tensors are released as soon as a batch is done.
                del inputs, outputs, logits, ids, types, mask
        gc.collect()
        return [r if r is not None else list(neutral) for r in results]

    def score_batch(self, premises: Sequence[str], hypothesis: str) -> List[float]:
        """ P(entailment) for each premise (see predict_probs). """
        return [p[0] for p in self.predict_probs(premises, hypothesis)]

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """ P(entailment) of a single premise/hypothesis pair (max over windows for a long premise). """
        return self.score_batch([premise], hypothesis)[0]


class FactChecker(object):
    """Fact checker base type"""

    def predict(self, fact: str, passages: List[dict]) -> str:
        """Predict S or NS for a fact and its retrieved evidence."""
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


# --- PART 1: WORD OVERLAP CLASSIFIER ---
class WordRecallThresholdFactChecker(FactChecker):
    """Predict S when the best passage recall reaches the configured threshold."""
    PREPROCESSING = {"lowercase": True, "remove_stopwords": True, "normalizer": "porter", "min_token_len": 2}
    METRIC = "recall"
    AGGREGATION = "max"
    DROP_TITLE = True

    DEFAULT_THRESHOLD = 0.5634920634920635

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        self.threshold = float(threshold)
        self.preprocessor = TextPreprocessor(**self.PREPROCESSING)

    def score(self, fact: str, passages: List[dict]) -> float:
        """ Word-overlap support score in [0, 1]; predict() thresholds it. """
        return word_overlap_score(fact, passages, self.preprocessor, self.METRIC, self.AGGREGATION, self.DROP_TITLE)

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Equality is supported; tuning uses exactly the same comparison.
        # A zero score means no lexical evidence, even at a threshold of zero.
        value = self.score(fact, passages)
        return "S" if value > 0.0 and value >= self.threshold else "NS"


# --- PART 2: ENTAILMENT CLASSIFIER AND PRUNING ---
class EntailmentFactChecker(FactChecker):
    """Prune low-overlap facts, then threshold the maximum evidence entailment probability."""
    DEFAULT_THRESHOLD = 0.2535
    # --- PRUNING SETTINGS: 0.0 disables the corresponding level ---
    FACT_PRUNE_THRESHOLD = 0.30
    PRUNE_METRIC = "recall"
    EVIDENCE_PRUNE_THRESHOLD = 0.0
    # Evidence-pruning safeguards (used only when its threshold is positive).
    EVIDENCE_KEEP_TOP_K = 3
    EVIDENCE_KEEP_LEAD = True
    EVIDENCE_MODE = "context"
    # Add pronoun-resolved copies of sentences that start with a personal pronoun.
    TITLED_PRONOUN_UNITS = False
    # Evidence units with fewer words than this are dropped (0 disables it).
    MIN_EVIDENCE_WORDS = 0
    EARLY_EXIT = True

    EVIDENCE_MODES = ("sentence", "context", "passage")
    PRUNE_METRICS = ("recall", "fuzzy_recall", "idf_recall", "tfidf_cosine")
    UNIT_PRUNE_METRICS = ("recall", "fuzzy_recall")  # IDF needs the fact's whole passage set, not a single unit
    # Singular personal pronouns only: a sentence-initial "They" usually refers to several people.
    _PRONOUN_RE = re.compile(r"^(He|She|His|Her)\b")

    def __init__(self, ent_model, threshold: Optional[float] = None, fact_prune_threshold: Optional[float] = None,
                 evidence_prune_threshold: Optional[float] = None, prune_metric: Optional[str] = None,
                 evidence_keep_top_k: Optional[int] = None, evidence_keep_lead: Optional[bool] = None,
                 evidence_mode: Optional[str] = None, titled_pronoun_units: Optional[bool] = None,
                 min_evidence_words: Optional[int] = None, early_exit: Optional[bool] = None,
                 overlap_checker: Optional[FactChecker] = None):
        """Initialise configuration and validate supplied options."""
        def pick(value, default):
            return default if value is None else value

        self.ent_model = ent_model
        self.threshold = float(pick(threshold, self.DEFAULT_THRESHOLD))
        self.fact_prune_threshold = float(pick(fact_prune_threshold, self.FACT_PRUNE_THRESHOLD))
        self.evidence_prune_threshold = float(pick(evidence_prune_threshold, self.EVIDENCE_PRUNE_THRESHOLD))
        self.prune_metric = pick(prune_metric, self.PRUNE_METRIC)
        self.evidence_keep_top_k = int(pick(evidence_keep_top_k, self.EVIDENCE_KEEP_TOP_K))
        self.evidence_keep_lead = bool(pick(evidence_keep_lead, self.EVIDENCE_KEEP_LEAD))
        self.evidence_mode = pick(evidence_mode, self.EVIDENCE_MODE)
        self.titled_pronoun_units = bool(pick(titled_pronoun_units, self.TITLED_PRONOUN_UNITS))
        self.min_evidence_words = int(pick(min_evidence_words, self.MIN_EVIDENCE_WORDS))
        self.early_exit = bool(pick(early_exit, self.EARLY_EXIT))
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("threshold must be finite and in (0, 1].")
        for value in (self.fact_prune_threshold, self.evidence_prune_threshold):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("pruning thresholds must be in [0, 1].")
        if self.evidence_mode not in self.EVIDENCE_MODES:
            raise ValueError("evidence_mode must be one of " + repr(self.EVIDENCE_MODES))
        if self.prune_metric not in self.PRUNE_METRICS:
            raise ValueError("prune_metric must be one of " + repr(self.PRUNE_METRICS))
        if self.evidence_prune_threshold > 0 and self.prune_metric not in self.UNIT_PRUNE_METRICS:
            raise ValueError("evidence-level pruning needs prune_metric in " + repr(self.UNIT_PRUNE_METRICS))
        if self.evidence_keep_top_k < 0 or self.min_evidence_words < 0:
            raise ValueError("evidence_keep_top_k and min_evidence_words must be >= 0")
        # The Part 1 scorer (and its preprocessing cache, which makes the shared passages cheap) is reused.
        self.overlap_checker = overlap_checker if overlap_checker is not None else WordRecallThresholdFactChecker()
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.stats = Counter()

    @staticmethod
    def _clean_evidence(text: str) -> str:
        """ Removes wiki mark-up that carries no claim content: closed {{cite ...}} / {{citation ...}} templates
        (outermost only, so nested templates are handled), <ref> elements, other HTML tags, [[link|label]] syntax and
        external-link brackets; unescapes HTML entities and normalises whitespace. Unclosed templates are kept, so
        real prose after a broken template is never deleted. """
        ranges, stack = [], []
        for match in re.finditer(r"\{\{|\}\}", text):
            if match.group() == "{{":
                stack.append(match.start())
            elif stack:
                start = stack.pop()
                if re.match(r"\{\{\s*(?:cite\b|citation\b)", text[start:], re.I):
                    ranges.append((start, match.end()))
        for start, end in sorted(ranges, reverse=True):
            if not any(a <= start and end <= b and (a, b) != (start, end) for a, b in ranges):
                text = text[:start] + " " + text[end:]
        text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\[(?:https?://\S+)\s+([^\]]+)\]", r"\1", text)
        return " ".join(html.unescape(text).split())

    @staticmethod
    def _sentence_units(text: str, nlp, repair_quotes: bool = False) -> List[str]:
        """ Rule-based sentence split. With repair_quotes, a boundary found inside a double-quoted span (e.g. a film
        title ending in '.') is undone by merging the pieces until the quote is closed. """
        sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        if not repair_quotes:
            return sentences
        result, pending, quote = [], [], None
        for sentence in sentences:
            pending.append(sentence)
            for char in sentence:
                if char == '"':
                    quote = None if quote == '"' else ('"' if quote is None else quote)
                elif char == "“" and quote is None:
                    quote = "”"
                elif char == "”" and quote == "”":
                    quote = None
            if quote is None:
                result.append(" ".join(pending))
                pending = []
        if pending:
            result.append(" ".join(pending))
        return result

    @staticmethod
    def _entity_name(passages: List[dict]) -> str:
        """ Page title without a disambiguation suffix: "Don Featherstone (artist)" -> "Don Featherstone". """
        title = passages[0].get("title", "") if passages else ""
        return re.sub(r"\s*\([^)]*\)\s*$", "", title or "").strip()

    def _resolve_pronoun(self, sentence: str, name: str) -> Optional[str]:
        """ "She sang." -> "<name> sang."; "His album ..." -> "<name>'s album ...". None if nothing to resolve. """
        match = self._PRONOUN_RE.match(sentence)
        if not match or not name:
            return None
        possessive = match.group(1) in ("His", "Her")
        return name + ("'s" if possessive else "") + sentence[match.end():]

    def evidence_records(self, passages: List[dict], mode: Optional[str] = None,
                         titled: Optional[bool] = None) -> List[dict]:
        """
        All candidate evidence units of the passages, deduplicated, in document order.
        :param mode: "sentence", "context" or "passage" (default: self.evidence_mode)
        :param titled: add pronoun-resolved units (default: self.titled_pronoun_units); ignored in "passage" mode
        :return: list of {"text", "kind", "lead"}; kind is "sentence", "context" (previous + current sentence),
        "titled" (pronoun replaced by the entity name) or "passage"; lead marks the opening sentence of a Wikipedia
        article (and the units that contain it)
        """
        mode = self.evidence_mode if mode is None else mode
        titled = self.titled_pronoun_units if titled is None else titled
        name = self._entity_name(passages)
        records, seen = [], set()

        def add(text, kind, lead):
            if text and text not in seen:
                seen.add(text)
                records.append({"text": text, "kind": kind, "lead": lead})

        for passage in passages:
            segments = [self._clean_evidence(s) for s in re.split(r"</?s\s*>", passage.get("text") or "", flags=re.I)]
            segments = [s for s in segments if s]
            # Retrieved chunks that open a Wikipedia article start with the page title ("Lanny Flaherty Lanny
            # Flaherty (born ...) is an American actor."); only their first sentence is the article's lead sentence.
            title = (passage.get("title") or "").lower()
            opens_article = bool(segments) and bool(title) and segments[0].lower().startswith(title)
            if mode == "passage":
                add(" ".join(segments), "passage", opens_article)
                continue
            for segment_index, segment in enumerate(segments):
                sentences = self._sentence_units(segment, self.nlp, repair_quotes=True)
                for index, sentence in enumerate(sentences):
                    first_segment = opens_article and segment_index == 0
                    add(sentence, "sentence", first_segment and index == 0)
                    if mode == "context" and index:
                        add(sentences[index - 1] + " " + sentence, "context", first_segment and index == 1)
                    if titled:
                        resolved = self._resolve_pronoun(sentence, name)
                        if resolved:
                            add(resolved, "titled", False)
        return records

    def evidence_units(self, passages: List[dict], mode: Optional[str] = None) -> List[Tuple[str, str]]:
        """ (text, kind) of every candidate evidence unit (before pruning and filters). """
        return [(r["text"], r["kind"]) for r in self.evidence_records(passages, mode)]

    def iter_evidence(self, passages: List[dict]):
        """ Texts of all candidate evidence units (before pruning and filters). """
        for record in self.evidence_records(passages):
            yield record["text"]

    def word_overlap(self, fact: str, passages: List[dict], metric: Optional[str] = None) -> float:
        """ Word overlap of the fact with its best passage (Part 1 preprocessing, entity name removed). """
        return word_overlap_score(fact, passages, self.overlap_checker.preprocessor, metric or self.prune_metric,
                                  "max", True)

    def unit_overlap(self, fact: str, passages: List[dict], unit: str, metric: Optional[str] = None) -> float:
        """ The same word overlap computed against a single evidence unit (recall for the fact-level-only metrics). """
        if metric is None:
            metric = self.prune_metric if self.prune_metric in self.UNIT_PRUNE_METRICS else "recall"
        title = passages[0].get("title", "") if passages else ""
        return word_overlap_score(fact, [{"title": title, "text": unit}], self.overlap_checker.preprocessor,
                                  metric, "max", True)

    # --- EVIDENCE-LEVEL PRUNING ---
    @staticmethod
    def plan_evidence(records: List[dict], overlaps: List[float], min_words: int, evidence_threshold: float,
                      keep_top_k: int, keep_lead: bool) -> List[int]:
        """Filter and order evidence consistently for inference and cached experiments."""
        candidates = [i for i, r in enumerate(records)
                      if not (min_words and len(r["text"].split()) < min_words)]
        if evidence_threshold > 0:
            ranked = sorted(candidates, key=lambda i: (-overlaps[i], i))
            top = set(ranked[:keep_top_k])
            candidates = [i for i in candidates
                          if overlaps[i] >= evidence_threshold or i in top or (keep_lead and records[i]["lead"])]
        return sorted(candidates, key=lambda i: (-overlaps[i], i))

    def select_evidence(self, fact: str, passages: List[dict]) -> List[str]:
        """ Evidence texts that are scored for this fact (after evidence pruning and filters), in scoring order. """
        records = self.evidence_records(passages)
        overlaps = [self.unit_overlap(fact, passages, r["text"]) for r in records]
        order = self.plan_evidence(records, overlaps, self.min_evidence_words, self.evidence_prune_threshold,
                                   self.evidence_keep_top_k, self.evidence_keep_lead)
        return [records[i]["text"] for i in order]

    # --- FACT-LEVEL PRUNING ---
    def _prepare(self, fact: str, passages: List[dict]) -> List[str]:
        """ Applies fact-level pruning, then returns the evidence to score ([] = predict NS without the model). """
        self.stats["facts"] += 1
        if not fact or not fact.strip() or not passages:
            return []
        if self.fact_prune_threshold > 0 and self.word_overlap(fact, passages) < self.fact_prune_threshold:
            self.stats["pruned_facts"] += 1
            return []
        evidence = self.select_evidence(fact, passages)
        self.stats["candidate_units"] += len(evidence)
        return evidence

    def score_details(self, fact: str, passages: List[dict]) -> Tuple[float, str]:
        """Return maximum entailment probability and its evidence; pruned facts return zero."""
        evidence = self._prepare(fact, passages)
        if not evidence:
            return 0.0, ""
        scores = self.ent_model.score_batch(evidence, fact)
        self.stats["scored_units"] += len(evidence)
        best = max(range(len(scores)), key=lambda i: scores[i])
        return float(scores[best]), evidence[best]

    def score(self, fact: str, passages: List[dict]) -> float:
        return self.score_details(fact, passages)[0]

    def predict(self, fact: str, passages: List[dict]) -> str:
        if not self.early_exit:
            return "S" if self.score(fact, passages) >= self.threshold else "NS"
        evidence = self._prepare(fact, passages)
        chunk = max(1, int(getattr(self.ent_model, "batch_size", 16)))
        for start in range(0, len(evidence), chunk):
            part = evidence[start:start + chunk]
            self.stats["scored_units"] += len(part)
            if max(self.ent_model.score_batch(part, fact)) >= self.threshold:
                self.stats["early_exits"] += 1
                return "S"
        return "NS"


# --- OPTIONAL: HYBRID DEPENDENCY AND LEXICAL CLASSIFIER ---
class DependencyRecallThresholdFactChecker(FactChecker):
    """Optional hybrid of dependency recall and passage lexical recall, with structural guards."""

    DEFAULT_THRESHOLD = 0.5276785714285713
    DEFAULT_DEPENDENCY_WEIGHT = 0.1
    MODEL_NAME = "en_core_web_sm"
    _IGNORED_DEPS = frozenset({"punct", "ROOT", "root", "det", "case", "aux", "auxpass", "cop", "dep"})

    def __init__(self, threshold: Optional[float] = None, dependency_weight: Optional[float] = None):
        self.threshold = self.DEFAULT_THRESHOLD if threshold is None else float(threshold)
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("threshold must be finite and in (0, 1].")
        self.dependency_weight = (self.DEFAULT_DEPENDENCY_WEIGHT if dependency_weight is None
                                  else float(dependency_weight))
        if not math.isfinite(self.dependency_weight) or not 0 < self.dependency_weight <= 1:
            raise ValueError("dependency_weight must be finite and in (0, 1].")
        self.lexical_checker = WordRecallThresholdFactChecker()
        try:
            self.nlp = spacy.load(self.MODEL_NAME, disable=["ner"])
        except OSError as exc:
            raise RuntimeError(
                "Parsing mode needs the spaCy English parser. Install it with: "
                "python -m spacy download en_core_web_sm"
            ) from exc
        if not self.nlp.has_pipe("parser") or not self.nlp.has_pipe("lemmatizer"):
            raise RuntimeError("Parsing mode needs a dependency parser and lemmatizer.")
        self._cache = {}

    @staticmethod
    def _lemma(token) -> str:
        return (token.lemma_ or token.text).lower()

    def _features(self, tokens, entity_words):
        def word(token):
            if token.lower_ in {"he", "she", "his", "her"}:
                return "__entity__"
            if token.pos_ == "PROPN" and token.lower_ in entity_words:
                return "__entity__"
            return self._lemma(token)

        relations, content, polarity = set(), set(), {}
        for token in tokens:
            if token.is_space or token.is_punct:
                continue
            value = word(token)
            if not token.is_stop or token.like_num:
                content.add(value)
            if token.pos_ in {"VERB", "AUX", "ADJ"}:
                negative = any(child.dep_ == "neg" or child.lower_ in {"never", "not", "n't"}
                               for child in token.children)
                keys = {"lemma:" + self._lemma(token)}
                keys.update("surface:" + stem for stem in self.lexical_checker.preprocessor.process(token.text))
                for predicate in keys:
                    polarity.setdefault(predicate, set()).add(negative)
            if token.dep_ in self._IGNORED_DEPS or token.head == token:
                continue
            head = word(token.head)
            if head == value == "__entity__":
                continue
            relations.add((head, token.dep_, value))
        return relations, content, polarity

    def _analyses(self, text: str, entity_words=frozenset()):
        text = EntailmentFactChecker._clean_evidence(text or "")
        key = (text, tuple(sorted(entity_words)))
        if key not in self._cache:
            if not text:
                self._cache[key] = []
            else:
                doc = self.nlp(text)
                self._cache[key] = [(*self._features(sentence, entity_words), sentence.text)
                                    for sentence in doc.sents]
        return self._cache[key]

    def get_dependencies(self, sent: str):
        """Return the union of labelled lemma triples; no page-title substitution."""
        return set().union(*(row[0] for row in self._analyses(sent)))

    @staticmethod
    def _opposite_polarity(fact_polarity, evidence_polarity):
        return any(evidence_polarity.get(predicate) and values.isdisjoint(evidence_polarity[predicate])
                   for predicate, values in fact_polarity.items())

    @staticmethod
    def _reversed_arguments(fact_relations, evidence_relations):
        subjects = {"nsubj", "csubj"}
        objects = {"dobj", "obj"}
        for head, relation, subject in fact_relations:
            if relation not in subjects:
                continue
            for predicate, role, obj in fact_relations:
                if predicate != head or role not in objects or subject == obj:
                    continue
                if (any(h == head and r in subjects and word == obj for h, r, word in evidence_relations)
                        and any(h == head and r in objects and word == subject
                                for h, r, word in evidence_relations)):
                    return True
        return False

    def score_components(self, fact: str, passages: List[dict]) -> Tuple[float, float]:
        """Return dependency recall and lexical recall over structurally eligible evidence."""
        if not fact or not fact.strip() or not passages:
            return 0.0, 0.0
        name = EntailmentFactChecker._entity_name(passages)
        entity_words = frozenset(token.lower_ for token in self.nlp.make_doc(name) if token.is_alpha)
        fact_rows = self._analyses(fact, entity_words)
        if not fact_rows:
            return 0.0, 0.0
        relations = set().union(*(row[0] for row in fact_rows))
        content = set().union(*(row[1] for row in fact_rows))
        polarity = {}
        for row in fact_rows:
            for predicate, values in row[2].items():
                polarity.setdefault(predicate, set()).update(values)
        best, eligible_passages = 0.0, []
        for passage in passages:
            eligible_sentences = []
            for evidence, words, signs, text in self._analyses(passage.get("text", ""), entity_words):
                if (self._opposite_polarity(polarity, signs)
                        or self._reversed_arguments(relations, evidence)):
                    continue
                value = (len(relations & evidence) / len(relations) if relations else
                         len(content & words) / len(content) if content else 0.0)
                best = max(best, value)
                eligible_sentences.append(text)
            if eligible_sentences:
                eligible_passages.append({"title": passage.get("title", ""),
                                          "text": " ".join(eligible_sentences)})
        lexical = self.lexical_checker.score(fact, eligible_passages) if eligible_passages else 0.0
        return best, lexical

    def score(self, fact: str, passages: List[dict]) -> float:
        dependency, lexical = self.score_components(fact, passages)
        return self.dependency_weight * dependency + (1 - self.dependency_weight) * lexical

    def predict(self, fact: str, passages: List[dict]) -> str:
        value = self.score(fact, passages)
        return "S" if value > 0.0 and value >= self.threshold else "NS"
