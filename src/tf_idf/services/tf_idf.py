import nltk
import numpy as np

from tf_idf.services.sparse_matrix import SparseMatrix
from tf_idf.services.token_counter import TokensCounter


class TfIdfVectorizer:
    def __init__(self, ngrams: int = 1) -> None:
        self._vocabulary: dict[str, int] = {}
        self._feature_list: list[str] = []

        self._number_of_documents_in_corpus = 0
        self._number_of_features = 0

        self._ngrams = ngrams

        self._validate_params()

    def fit(self, texts: np.ndarray) -> None:
        self._number_of_documents_in_corpus = len(texts)

        tokenized_documents: list[list[str]] = [
            self._tokenize(text) for text in texts
        ]

        tokens_counter = TokensCounter(tokenized_documents)

        self._vocabulary = tokens_counter.count()
        self._number_of_features = len(self._vocabulary)
        self._feature_list = list(self._vocabulary.keys())

    def transform(self, texts: np.ndarray, init_document_id: int = 0) -> SparseMatrix:
        feature_indexes: list[tuple[int, int]] = []
        feature_tf_idfs: list[float] = []

        for document_id, document in enumerate(texts):
            tokenized_document = self._tokenize(document)
            document_features = set(tokenized_document)
            tmp_tf_idfs = []

            for feature in document_features:
                if feature not in self._feature_list:
                    continue

                feature_tf_idf = self._calculate_tf_idf(tokenized_document, feature)
                feature_result_matrix_idxes = (document_id + init_document_id, self._feature_list.index(feature))

                feature_indexes.append(feature_result_matrix_idxes)
                tmp_tf_idfs.append(feature_tf_idf)

            feature_tf_idfs.extend(self._normalize(tmp_tf_idfs))

        return SparseMatrix(
            indexes=feature_indexes,
            data=feature_tf_idfs,
            shape=(self._number_of_documents_in_corpus, self._number_of_features),
        )

    def get_feature_list(self) -> list[str]:
        return self._feature_list

    def _validate_params(self) -> None:
        if not isinstance(self._ngrams, int) or self._ngrams < 1:
            raise ValueError("N-grams parameter must be int higher or equal to 1")

    def _tokenize(self, txt: str) -> list[str]:
        tokens = nltk.word_tokenize(txt)
        preprocessed_tokens = self._preprocess_text(tokens)

        return self._combine_into_ngrams(preprocessed_tokens)

    def _preprocess_text(self, tokens: list[str]) -> list[str]:
        preprocessed_tokens: list[str] = []

        for token in tokens:
            if not token.isalnum():
                continue

            token = token.lower()

            preprocessed_tokens.append(token)

        return preprocessed_tokens

    def _combine_into_ngrams(self, tokens: list[str]) -> list[str]:
        if self._ngrams == 1:
            return tokens

        ngrams_combined: list[str] = []

        for idx in range(len(tokens) - self._ngrams + 1):
            ngram = " ".join(tokens[idx:idx + self._ngrams])
            ngrams_combined.append(ngram)

        return ngrams_combined

    def _calculate_tf_idf(self, document: list[str], gram: str) -> float:
        return self._calculate_tf(document, gram) * self._calculate_idf(gram)

    def _calculate_tf(self, document: list[str], gram: str) -> float:
        return document.count(gram) / len(document)

    def _calculate_idf(self, gram: str) -> float:
        return np.log(self._number_of_documents_in_corpus / (self._vocabulary.get(gram, 0)+1)) + 1

    def _normalize(self, vector: list[float]) -> list[float]:
        vector = np.array(vector)
        absolute_value = np.sqrt((vector**2).sum())

        return list(vector / absolute_value)
