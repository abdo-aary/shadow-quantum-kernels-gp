from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

from typing import Sequence

import numpy as np

from src.circuits.configs import CircuitArchitecture
from src.runners.circuit_running import Results

from qiskit.quantum_info import SparseObservable


class BaseFeatureMapsRetriever(ABC):
    """
    Base class for retrieving feature maps
    """
    arch: CircuitArchitecture
    fmps: np.ndarray = None
    observables: Sequence[SparseObservable]

    @abstractmethod
    def get_feature_maps(self, results: Results):
        ...

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save the MKLearner (including any precomputed approximations) to disk.

        Parameters
        ----------
        path : str
            Destination path for the pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseFeatureMapsRetriever":
        """
        Load a previously saved BaseFeatureMapsRetriever from disk.

        Parameters
        ----------
        path : str
            Path to the pickle file created by :meth:`save`.

        Returns
        -------
        MKLearner
            The reconstructed learner instance.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BaseFeatureMapsRetriever):
            raise TypeError(f"Expected a BaseFeatureMapsRetriever instance in {path!r}, got {type(obj)!r}.")
        return obj
