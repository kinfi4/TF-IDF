import numpy as np


class SparseMatrix:
    def __init__(
        self,
        shape: tuple[int, ...],
        indexes: list[tuple[int, ...]] = None,
        data: list[int | float | str | bool] = None,
    ) -> None:
        if data is None or indexes is None:
            data, indexes = [], []

        self._data = data
        self._indexes = indexes
        self._shape = shape

        self._validate_matrix_shape()

    def to_numpy(self) -> np.ndarray:
        result_matrix = np.zeros(shape=self._shape)

        for index, value in zip(self._indexes, self._data):
            result_matrix[index] = value

        return result_matrix

    def _validate_matrix_shape(self) -> None:
        if len(self._indexes) > 0 and len(self._shape) != len(self._indexes[0]):
            raise ValueError("[SparseMatrix] Matrix shape must be the same dimension that indexes.")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def add_values(self, indexes: list[tuple[int, ...]], values: list[int | float | str | bool]) -> None:
        self._indexes.extend(indexes)
        self._data.extend(values)

    def get_indexes(self) -> list[tuple[int, ...]]:
        return self._indexes

    def get_values(self) -> list[int | float | str | bool]:
        return self._data

    def set_indexes(self, indexes: list[tuple[int, ...]]) -> None:
        self._indexes = indexes
