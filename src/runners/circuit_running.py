from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
from qiskit.providers import BackendV2

import numpy as np
from qiskit import QuantumCircuit

from qiskit_aer import AerSimulator
from qiskit import transpile

from src.circuits.configs import CircuitArchitecture

DEFAULT_BACKEND = AerSimulator(
    method="statevector",  # CPU statevector
    device="CPU"
)


class Results(ABC):
    ...

@dataclass
class ExactResults:
    """
    This must contain exact results which are to be exact statevectors obtained after running each circuit in each pub.
    Concretely, having a list of pubs = [pub_r : 1 <= r <= R], where each pub_r = (qc, params_r) with params_r is a
    matrix of shape (M, num_params), results must be an array of shape (R, M, 2**num_qubits), which stores the
    different states prepared after running each circuit with a non-data parameterization_r and a data x_i : 1<=i<=M.
    """
    states: np.ndarray
    arch: CircuitArchitecture

class BaseCircuitsRunner(ABC):
    """
    Interface setting the logic of how we run the circuits. This is to be implemented by two classes, ExactCircuitRunner
    and ShadowCircuitRunner. The results of these are to be of the shape Results.
    """

    @abstractmethod
    def run_pubs(self, **kwargs) -> Results:
        ...

class ExactCircuitsRunner(BaseCircuitsRunner):
    """
    Interface setting the logic of how we run the circuits. This is to be implemented by two classes, ExactCircuitRunner
    and ShadowCircuitRunner. The results of these
    """
    def __init__(self, arch: CircuitArchitecture):
        self.arch = arch

    def run_pubs(
            self,
            pubs: List[Tuple[QuantumCircuit, np.ndarray]],
            max_threads: Optional[BackendV2] = None,
            backend: Optional[BackendV2] = None,
    ) -> ExactResults:
        """
        #     This method must compute all (R, M) states in a parallel fashion
        #     """

        backend = backend or DEFAULT_BACKEND

        # --- basic shapes ---
        R = len(pubs)
        if R == 0:
            raise ValueError("pubs must be non-empty")

        # Assume all circuits have same #qubits and same M
        n_qubits = pubs[0][0].num_qubits
        M = pubs[0][1].shape[0]
        dim = 1 << n_qubits  # 2**n_qubits

        # --- configure Aer parallelism ---
        if max_threads is None or max_threads < 1:
            max_threads = 0  # 0 => "use all available cores"

        backend.set_options(
            method="statevector",
            device="CPU",
            max_parallel_threads=max_threads,
            max_parallel_experiments=0,  # up to max_parallel_threads
            max_parallel_shots=1,  # you don't need shot parallelism here
            # ensure we actually use multithreading for relatively small circuits
            statevector_parallel_threshold=max(10, n_qubits // 2),
        )

        all_circs = []
        index_map = []  # to map back from flat index -> (r, m)

        # --- build bound circuits once, with save_statevector ---
        for r, (qc, params_r) in enumerate(pubs):
            if qc.num_qubits != n_qubits:
                raise ValueError("All circuits must have same number of qubits")

            if params_r.shape[0] != M:
                raise ValueError("All pubs must have the same M (rows in params_r)")

            # ensure save_statevector() at end
            if not any(
                    getattr(inst.operation, "name", "") in ("save_state", "save_statevector")
                    for inst in qc.data
            ):
                base_qc = qc.copy()
                base_qc.save_statevector()
            else:
                base_qc = qc

            # transpile *once* per pub
            tqc = transpile(base_qc, backend=backend)

            param_order = list(tqc.parameters)  # ordered list of Parameter objects
            num_params = len(param_order)
            if params_r.shape[1] != num_params:
                raise ValueError(
                    f"params_r has {params_r.shape[1]} columns but circuit "
                    f"has {num_params} parameters"
                )

            for m in range(M):
                values = params_r[m]
                bind_dict = dict(zip(param_order, values))
                bound_circ = tqc.assign_parameters(bind_dict, inplace=False)
                all_circs.append(bound_circ)
                index_map.append((r, m))

        # --- run everything in one go; Aer handles internal parallelism ---
        job = backend.run(all_circs)
        result = job.result()

        # --- extract statevectors and pack into (R, M, 2**n) ---
        states = np.empty((R, M, dim), dtype=complex)

        for k, (r, m) in enumerate(index_map):
            # For statevector method + save_statevector(),
            # Aer stores state as result.data(k)["statevector"]
            data_k = result.data(k)
            sv = np.array(data_k["statevector"], dtype=complex)
            if sv.size != dim:
                raise RuntimeError(
                    f"Unexpected statevector size {sv.size}, expected {dim}"
                )
            states[r, m, :] = sv

        return ExactResults(states=states, arch=self.arch)
