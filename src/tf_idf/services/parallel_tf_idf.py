import multiprocessing as mp
from multiprocessing.pool import AsyncResult

import numpy as np

from tf_idf.services.tf_idf import TfIdfVectorizer
from tf_idf.services.sparse_matrix import SparseMatrix


class ParallelTfIdfVectorizer(TfIdfVectorizer):
    def __init__(self, workers_number: int = None, ngrams: int = 1) -> None:
        super().__init__(ngrams)

        if workers_number is None:
            workers_number = mp.cpu_count()

        self._workers_number = workers_number

    def transform(self, texts: np.ndarray, init_document_id: int = 0) -> SparseMatrix:
        pool = mp.Pool(self._workers_number)

        texts_sub_list = np.array_split(texts, self._workers_number)

        transform_results: list[AsyncResult] = []
        array_idx = 0

        for sub_array in texts_sub_list:
            transform_results.append(pool.apply_async(self._transform_sub_array, (sub_array, array_idx)))

            array_idx = array_idx + len(sub_array)

        result_matrix = SparseMatrix(shape=(self._number_of_documents_in_corpus, self._number_of_features))
        for future in transform_results:
            matrix = future.get()
            result_matrix = self._merge_sparce_matrixes(result_matrix, matrix)

        return result_matrix

    def _merge_sparce_matrixes(self, matrix1: SparseMatrix, matrix2: SparseMatrix) -> SparseMatrix:
        matrix1.add_values(matrix2.get_indexes(), matrix2.get_values())

        return matrix1

    def _transform_sub_array(self, texts: np.ndarray, initial_idx: int) -> SparseMatrix:
        return super().transform(texts, initial_idx)
