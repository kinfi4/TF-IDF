class TokensCounter:
    def __init__(self, corpus: list[list[str]]) -> None:
        self._corpus = corpus

    def count(self) -> dict[str, int]:
        _result_dict: dict[str, int] = {}

        for document in self._corpus:
            for gram in set(document):
                if gram not in _result_dict:
                    _result_dict[gram] = 0

                _result_dict[gram] += 1

        return _result_dict
