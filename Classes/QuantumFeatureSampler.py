import numpy as np
import math
from qiskit_script import *
from qiskit import QuantumCircuit, transpile
import qiskit_script
try:
    from qiskit_aer.primitives import Sampler as AerSampler
except ImportError:
    AerSampler = None

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QISKIT_RUNTIME_AVAILABLE = True
except ImportError:
    QISKIT_RUNTIME_AVAILABLE = False

class QuantumFeatureSampler:
    """
    Returns per-qubit marginal P(1) vector using Aer Simulator (training) or IBM Runtime (evaluation).
    - training: uses Aer (fast)
    - evaluation: tries IBM Runtime SamplerV2 (QPU) when prefer_qpu=True
    """
    def __init__(self, n_qubits:int, shots:int=256, train_mode:str="simulator", eval_mode:str="qpu", ibm_backend:str=None):
        self.n_qubits = n_qubits
        self.shots = shots
        self.train_mode = train_mode
        self.eval_mode = eval_mode
        self.ibm_backend = ibm_backend
        # local aer sampler
        if AerSampler is not None:
            try:
                self.aer_sampler = AerSampler(options={"shots": shots})
            except Exception:
                self.aer_sampler = None
        else:
            self.aer_sampler = None
        # runtime availability
        self.QISKIT_RUNTIME_AVAILABLE = QISKIT_RUNTIME_AVAILABLE

    # def _make_simple_pqc(self):
    #     from qiskit import QuantumCircuit
    #     qc = QuantumCircuit(self.n_qubits, self.n_qubits)
    #     for q in range(self.n_qubits):
    #         qc.h(q)
    #     qc.measure_all()
    #     return qc
    
    def _make_simple_pqc(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
       
         # 1. Feature Map Initialization
        for q in range(self.n_qubits):
             qc.h(q)
           
         # 2. Entanglement Layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
           
        qc.measure_all()
        return qc
    
    def _aer_marginals(self, qc):
        if self.aer_sampler is None:
            # fallback deterministic uniform marginals
            return 0.5 * np.ones(self.n_qubits, dtype=float)
        job = self.aer_sampler.run([qc])
        res = job.result()
        data = res[0].data
        counts = {}
        if hasattr(data, "meas") and data.meas is not None:
            counts = data.meas.get_counts()
        else:
            counts = data.get("counts", {})
        total = sum(counts.values()) if isinstance(counts, dict) else 0
        p1 = np.zeros(self.n_qubits, dtype=float)
        for bs, c in counts.items():
            bits = bs[::-1]
            for i, bit in enumerate(bits):
                if bit == '1':
                    p1[i] += c
        if total > 0:
            p1 /= total
        return p1

    def _qpu_marginals(self, qc, backend_name=None, shots=None):
        """
        Submit `qc` to IBM Runtime SamplerV2 and return (p1_vector, job_id).
        Blocks until result (small smoke test). Raises on failure.
        """
        if not self.QISKIT_RUNTIME_AVAILABLE:
            raise RuntimeError("IBM Runtime not available in this environment.")
        shots = shots or self.shots
        backend_name = backend_name or self.ibm_backend
        try:
            service = QiskitRuntimeService()
        except Exception as e:
            raise RuntimeError(f"Failed to create QiskitRuntimeService: {e}") from e

        # choose backend if not provided
        chosen_backend = backend_name
        if chosen_backend is None:
            try:
                chosen_backend = service.least_busy(operational=True, simulator=False).name
            except Exception:
                chosen_backend = None

        from qiskit_ibm_runtime import Session, SamplerV2
        try:
            with Session(service=service, backend=chosen_backend) as sess:
                sampler = SamplerV2(session=sess, options={"shots": shots})
                job = sampler.run([qc])
                job_id = getattr(job, "job_id", None)
                # block for smoke test (we prefer this for immediate feedback)
                res = job.result()
                dat = res[0].data
                counts = {}
                if hasattr(dat, "meas") and dat.meas is not None:
                    counts = dat.meas.get_counts()
                else:
                    counts = dat.get("counts", {})
                total = sum(counts.values()) if isinstance(counts, dict) else 0
                p1 = np.zeros(self.n_qubits, dtype=float)
                for bs, c in counts.items():
                    bits = bs[::-1]
                    for i, bit in enumerate(bits):
                        if bit == '1':
                            p1[i] += c
                if total > 0:
                    p1 /= total
                return p1, job_id
        except Exception as e:
            raise RuntimeError(f"Runtime sampler call failed: {e}") from e

    def run(self, controls: np.ndarray, training:bool=True, prefer_qpu:bool=False, qpu_backend: str = None, shots: int = None) -> np.ndarray:
        """
        controls: (B, d)
        returns: (B, n_qubits)
        - training=True -> use Aer
        - training=False & prefer_qpu=True -> try QPU (SamplerV2), fallback to Aer
        """
        B = int(controls.shape[0])
        qc = self._make_simple_pqc()
        if prefer_qpu:
            try:
                p1, job_id = self._qpu_marginals(qc, backend_name=qpu_backend, shots=shots)
                print(f"[QPU] job_id={job_id} (marginals returned).")
                return np.tile(p1[None, :], (B, 1))
            except Exception as e:
                print("[WARN] QPU call failed, falling back to Aer:", e)
                p1 = self._aer_marginals(qc)
                return np.tile(p1[None, :], (B, 1))
        p1 = self._aer_marginals(qc)
        return np.tile(p1[None,:], (B,1))