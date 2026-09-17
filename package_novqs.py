#============================================================
# This is the package for the paper titled 
# "Nonorthogonal variational quantum simulation for quantum chemistry"
#============================================================

import jax
print(f"JAX device: {jax.devices()}")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from functools import partial
import numpy as np

import pennylane as qml

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner
from pyscf import gto, scf, ci, cc, mcscf, fci

import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import os



class fSimAnsatz:
    def __init__(self, Nq, n_blocks, n_layers):
        self.Nq = Nq
        self.n_blocks = n_blocks
        self.n_layers = n_layers

    @staticmethod
    def fSim_gate(theta, phi, wires):
        """Define the fSim gate"""
        qml.IsingXY(2 * theta, wires=wires)
        qml.ControlledPhaseShift(phi, wires=wires)
        qml.SWAP(wires=wires)

    def apply_ansatz(self, params):
        """Ansatz"""
        idx = 0
        for b in range(self.n_blocks):
            # --- Start a block ---
            for l in range(self.n_layers):
                # Odd sublayer: (0,1), (2,3)...
                for i in range(0, self.Nq - 1, 2):
                    self.fSim_gate(params[idx], params[idx+1], wires=[i, i+1])
                    idx += 2
                # Even sublayer: (1,2), (3,4)...
                for i in range(1, self.Nq - 1, 2):
                    self.fSim_gate(params[idx], params[idx+1], wires=[i, i+1])
                    idx += 2
            
            # --- Add a layer of single-qubit Rz gates at the end of the block ---
            for i in range(self.Nq):
                qml.RZ(params[idx], wires=i)
                idx += 1



def init_params(Nq, n_blocks, n_layers, Ns, method, seed):
    # Precompute the number of parameters
    num_even_pairs = Nq // 2
    num_odd_pairs = (Nq - 1) // 2
    num_pairs_per_layer = num_even_pairs + num_odd_pairs
    total_fsim_gates = n_blocks * n_layers * num_pairs_per_layer 
    total_rz_gates = n_blocks * Nq
    n_params = Ns * (2 * total_fsim_gates + total_rz_gates)  # Two parameters per fSim gate
    """
    init_method: 'Zeros', 'Perturbation', 'Uniform', 'Gaussian'
    """
    key = jax.random.PRNGKey(seed)
    if method == 'Zeros':
        return jnp.zeros(n_params)
    elif method == 'Perturbation':
        delta = jnp.pi / 50
        return jax.random.uniform(key, shape=(n_params,), minval=-delta, maxval=delta)
    elif method == 'Uniform':
        delta = 1.0
        return jax.random.uniform(key, shape=(n_params,), minval=-delta, maxval=delta)
    elif method == 'Gaussian':
        std = 1.0
        return 1.0 * jax.random.normal(key, shape=(n_params,))
    else:
        raise ValueError(f"Unknown init method: {method}")



def get_circuit(dev, ansatz_obj, init_state='HF', hf_state=None):
    """
    Factory function that returns a configured QNode
    """
    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(params):
        # Prepare the initial state
        if init_state == 'US':
            for i in range(ansatz_obj.Nq):
                qml.Hadamard(wires=i)  # |+>^N
        elif init_state == 'HF':
            if hf_state is None:
                raise ValueError("HF state must be provided for 'HF' init_state")
            if ansatz_obj.n_blocks % 2 == 0:
                qml.BasisState(hf_state, wires=range(ansatz_obj.Nq))
            else:
                qml.BasisState(hf_state, wires=list(reversed(range(ansatz_obj.Nq))))
        
        ansatz_obj.apply_ansatz(params)
        return qml.state()
    
    return circuit



class VariationalEvolver:
    def __init__(self, circuit, hamiltonian_matrix, reg=1e-8):
        """
        Args:
            circuit: A function (QNode) that takes params and returns a state vector
            hamiltonian_matrix: The system Hamiltonian matrix (jax.numpy array)
            reg: Regularization coefficient used for matrix inversion
        """
        self.circuit = circuit
        self.H_mat = hamiltonian_matrix
        self.reg = reg
        # Predefine the Jacobian function
        self.jac_fn = jax.jacfwd(self.circuit)

    def _compute_M(self, dpsi):
        """Compute the variational metric tensor / Gram matrix"""
        M = jnp.real(dpsi.conj().T @ dpsi)
        return M

    def _compute_V(self, dpsi, psi):
        """Compute the force vector"""
        Hpsi = self.H_mat @ psi
        V = dpsi.conj().T @ Hpsi
        return V

    def _get_psi_and_jacobian(self, params):
        """Efficiently compute psi and dpsi together"""
        # params.size is the total number of parameters
        state, f_lin = jax.linearize(self.circuit, params)
        
        # Use the identity matrix to provide the standard basis vectors
        basis = jnp.eye(params.size)
        
        # Apply vmap to f_lin to obtain the Jacobian
        # f_lin(basis_vector) returns one column of the Jacobian
        jacobian = jax.vmap(f_lin)(basis) # Shape: (n_params, state_dim)
        
        return state, jacobian.T  # (state_dim,) (state_dim, n_params)
    
    @partial(jax.jit, static_argnums=(0,))
    def theta_dot_rte(self, t, params):
        """
        Real-time evolution equation: M @ theta_dot = Im(V)
        """
        psi, dpsi = self._get_psi_and_jacobian(params) 

        M = self._compute_M(dpsi)
        M += self.reg * jnp.eye(len(params))  # regularization
        
        V = self._compute_V(dpsi, psi)
        return jnp.linalg.solve(M, jnp.imag(V))
    
    @partial(jax.jit, static_argnums=(0,))
    def theta_dot_ite(self, t, params):
        """
        Imaginary-time evolution equation: M @ theta_dot = -Re(V)
        """
        psi, dpsi = self._get_psi_and_jacobian(params)

        M = jnp.real(dpsi.conj().T @ dpsi)
        M += self.reg * jnp.eye(len(params))
        
        V = self._compute_V(dpsi, psi)
        
        return jnp.linalg.solve(M, -jnp.real(V))



# Standalone numerical integrators (can also be placed in a separate utils.py file)
def rk4_step(f, t, y, dt):
    """General-purpose RK4 integrator"""
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def euler_step(f, t, y, dt):
    k1 = f(t, y)
    return y + dt * k1



class exactRTE:
    def __init__(self, eigvals, eigvecs, eig_coeffs):
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.eig_coeffs = eig_coeffs

    def get_exact_state(self, t):
        # V {e^{-iDt} (V^\dagger \psi_0)}
        phases = jnp.exp(-1j * self.eigvals * t)
        return self.eigvecs @ (phases * self.eig_coeffs)



class FidelityCalculator:
    """Class for calculating quantum-state fidelity"""
    def __init__(self, circuit, exact_solver):
        """
        Args:
            circuit: Quantum circuit function
            exact_solver: An instance of exactRTE
        """
        self.circuit = circuit
        self.exact_solver = exact_solver
        
        # Set up the vmapped function. Note: vmap is applied to the JIT-compiled calc_fidelity
        self._vmapped_fidelity = jax.vmap(self.calc_fidelity, in_axes=(0, 0))  # params[i], t[i]

    @partial(jax.jit, static_argnums=(0,))
    def calc_fidelity(self, params, t):
        """Compute fidelity at a single time point"""
        # 1. Get the variational quantum state
        psi_vqs = self.circuit(params)
        
        # 2. Call the exact_solver instance method to get the exact state
        # Ensure the method is called through self.exact_solver
        psi_exact = self.exact_solver.get_exact_state(t)
        
        # 3. Compute the squared overlap |<exact|vqs>|^2
        overlap = jnp.vdot(psi_exact, psi_vqs)
        return jnp.abs(overlap) ** 2

    def compute_batched(self, params_all, time_points, batch_size=100):
        """Compute fidelities in batches to avoid running out of GPU memory"""
        # Assume params_all has shape (T, n_params)
        T = params_all.shape[0]
        fidelities = []
        n_batches = math.ceil(T / batch_size)

        for i in tqdm(range(0, T, batch_size), total=n_batches, desc="Computing Fidelity"):
            # Get the current batch
            p_batch = params_all[i : i + batch_size]
            t_batch = time_points[i : i + batch_size]
            
            # Run the parallel calculation
            f = self._vmapped_fidelity(p_batch, t_batch)
            fidelities.append(f)

        return jnp.concatenate(fidelities, axis=0)



class exactITE:
    """Class for the analytical solution of imaginary-time evolution (ITE)"""
    def __init__(self, eigvals, eigvecs, eig_coeffs):
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.eig_coeffs = eig_coeffs
        self.E0 = eigvals[0]

    @partial(jax.jit, static_argnums=(0,))
    def get_exact_state(self, t):
        # V {e^{-Dt} (V^\dagger \psi_0)}
        shifted_eigvals = self.eigvals - self.E0  # shift the energy zero to avoid exponential overflow
        weights = jnp.exp(-self.shifted_eigvals * t)
        return self.eigvecs @ (weights * self.eig_coeffs)



class EnergyCalculator:
    """Class for comparing VQS and exact ITE energies"""
    def __init__(self, circuit, exact_solver, hamiltonian_matrix):
        """
        Args:
            circuit: Quantum circuit function
            exact_solver: An instance of exactITE
            hamiltonian_matrix: The system Hamiltonian matrix
        """
        self.circuit = circuit
        self.exact_solver = exact_solver
        self.H_mat = hamiltonian_matrix
        
        # Set up the vmapped function
        self._vmapped_energy = jax.vmap(self.calc_energy, in_axes=(0, 0))

    @partial(jax.jit, static_argnums=(0,))
    def _get_expectation(self, psi):
        """Compute the energy expectation value: <psi|H|psi> / <psi|psi>"""
        # Normalize the expectation value because ITE does not preserve the state norm
        expectation_val = jnp.vdot(psi, self.H_mat @ psi).real
        norm_sq = jnp.vdot(psi, psi).real
        return expectation_val / norm_sq

    @partial(jax.jit, static_argnums=(0,))
    def calc_energy(self, params, t):
        """Compute the energies of both states at a given time"""
        # 1. Compute the variational-state energy
        psi_vqs = self.circuit(params)
        E_vqs = self._get_expectation(psi_vqs)
        
        # 2. Compute the exact ITE state energy
        psi_exact = self.exact_solver.get_exact_state(t)
        E_ite = self._get_expectation(psi_exact)
        
        return E_vqs, E_ite

    def compute_batched(self, params_all, time_points, batch_size=100):
        """Compute the energy data in batches"""
        T = params_all.shape[0]
        e_vqs_list = []
        e_ite_list = []
        n_batches = math.ceil(T / batch_size)

        for i in tqdm(range(0, T, batch_size), total=n_batches, desc="Computing Energy"):
            p_batch = params_all[i : i + batch_size]
            t_batch = time_points[i : i + batch_size]
            
            # vmap returns two arrays of shape (batch_size,)
            e_vqs, e_ite = self._vmapped_energy(p_batch, t_batch)
            
            e_vqs_list.append(e_vqs)
            e_ite_list.append(e_ite)

        # Concatenate the results
        return jnp.concatenate(e_vqs_list, axis=0), jnp.concatenate(e_ite_list, axis=0)




class MultiStateEvolver:
    def __init__(self, circuit, eigvals, eigvecs, Nq, Ns, reg=1e-8):
        """
        Args:
            circuit: A single ansatz circuit function
            eigvals: Hamiltonian eigenvalues
            eigvecs: Matrix of Hamiltonian eigenvectors
            Nq: Number of qubits
            Ns: Number of component states in the superposition
            reg: Regularization coefficient
        """
        self.circuit = circuit
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.Nq = Nq
        self.Ns = Ns
        self.reg = reg
        
        # Set up the batched Jacobian function (vmap + jit)
        # Compute the forward pass and derivatives for Ns circuits in parallel
        self._circuits_jac_fn = jax.jit(jax.vmap(self._forward_and_jac))

    def _forward_and_jac(self, params_single):
        """Compute the state and Jacobian for a single parameter set"""
        state, f_lin = jax.linearize(self.circuit, params_single)
        basis = jnp.eye(params_single.size)
        jacobian = jax.vmap(f_lin)(basis)
        return state, jacobian.T  # (dim,), (dim, n_params)

    def _compute_M_V(self, states, dstates, params_coeffs):
        """Compute the metric matrix M and force vector V for the superposition state"""
        Ns = self.Ns
        coeffs = params_coeffs[-2*Ns:]
        coeffs_a = coeffs[:Ns] 
        coeffs_b = coeffs[-Ns:]
        coeffs_c = coeffs_a * jnp.exp(1j * coeffs_b)
        
        psi = coeffs_c @ states   # (dim,)
        norm_sq = jnp.vdot(psi, psi).real
        norm = jnp.sqrt(norm_sq)
        eig_coeffs = self.eigvecs.conj().T @ psi
        probs = jnp.abs(eig_coeffs)**2
        energy_unnormalized = jnp.sum(self.eigvals * probs)   # E = <psi| H |psi> = sum_n {E_n |<n|psi>|^2}

        dstates = coeffs_c[:, None, None] * dstates  # \partial c_i phi_i / \partial theta_ij = c_i \partial phi_i / \partial theta_ij 
        dstates = dstates.transpose(1, 0, 2).reshape(2**self.Nq, -1)  # dstates: (Ns, dim, n_params) -> (dim, Ns * n_params)
        dcoeffs_a = jnp.exp(1j * coeffs_b)[:, None] * states                    # (Ns, dim)
        dcoeffs_b = 1j * coeffs_c[:, None] * states  # (Ns, dim)
        dpsi = jnp.concatenate([dstates, dcoeffs_a.T, dcoeffs_b.T], axis=1)    # (dim, Ns * n_params + 2*Ns)
        dpsi_psi = dpsi.conj().T @ psi   # (Ns * n_params + 2*Ns,)

        M = jnp.real( (dpsi.conj().T @ dpsi) / norm_sq - jnp.outer(dpsi_psi, dpsi_psi.conj()) / norm_sq**2 ) 
        M += self.reg * jnp.eye(len(params_coeffs))  # regularization if shots is not None
        
        Hpsi = self.eigvecs @ (self.eigvals * eig_coeffs)  # H_mat @ psi
        V = (dpsi.conj().T @ Hpsi) / norm_sq - (dpsi_psi * energy_unnormalized) / (norm_sq**2) 
        
        return M, V

    @partial(jax.jit, static_argnums=(0,))
    def theta_dot_rte(self, t, params_coeffs):
        """Real-time evolution dynamics"""
        params = params_coeffs[:-2*self.Ns].reshape(self.Ns, -1)
        coeffs = params_coeffs[-2*self.Ns:]

        states, dstates = self._circuits_jac_fn(params)

        M, V = self._compute_M_V(states, dstates, params_coeffs)
        return jnp.linalg.solve(M, jnp.imag(V))

    @partial(jax.jit, static_argnums=(0,))
    def theta_dot_ite(self, t, params_coeffs):
        """Imaginary-time evolution dynamics"""
        params = params_coeffs[:-2*self.Ns].reshape(self.Ns, -1)
        coeffs = params_coeffs[-2*self.Ns:]

        states, dstates = self._circuits_jac_fn(params)

        M, V = self._compute_M_V(states, dstates, params_coeffs)
        return jnp.linalg.solve(M, -jnp.real(V))




class MultiStateFidelityCalculator:
    """Class for calculating fidelity between a variational superposition and the exact state"""
    def __init__(self, circuit, exact_solver, Ns):
        """
        Args:
            circuit_fn: A single ansatz circuit function
            exact_solver: An instance of ExactRTE or ExactITE
            Ns: Number of component states in the superposition
        """
        self.circuit = circuit
        self.exact_solver = exact_solver
        self.Ns = Ns
        
        # Predefine a batched circuit function to compute all Ns component states at once
        self.circuits_fn = jax.jit(jax.vmap(self.circuit, in_axes=(0,)))   # for multi ansatze
        # Set up the batched fidelity calculation function
        self._vmapped_fidelity = jax.vmap(self.calc_fidelity, in_axes=(0, 0))  # params_coeffs[i], t[i]

    @partial(jax.jit, static_argnums=(0,))
    def _get_vqs_state(self, params_coeffs):
        """Internal method that constructs a normalized superposition from the parameters"""
        # 1. Slice the array to extract circuit parameters and coefficients
        # params_coeffs layout: [theta_1, ..., theta_n, a_1, ..., a_Ns, b_1, ..., b_Ns]
        theta = params_coeffs[:-2*self.Ns].reshape(self.Ns, -1)
        coeffs_a = params_coeffs[-2*self.Ns : -self.Ns]
        coeffs_b = params_coeffs[-self.Ns :]
        
        # 2. Compute all component states (Ns, dim)
        states = self.circuits_fn(theta)
        
        # 3. Form the linear superposition with c_i = a_i * exp(i * b_i)
        coeffs_c = coeffs_a * jnp.exp(1j * coeffs_b)
        psi = coeffs_c @ states  # Form the superposition: sum_i c_i |phi_i>
        
        # 4. Normalize
        norm = jnp.linalg.norm(psi)
        return psi / norm

    @partial(jax.jit, static_argnums=(0,))
    def calc_fidelity(self, params_coeffs, t):
        """Compute fidelity at a single time point"""
        # 1. Get the normalized variational superposition
        psi_vqs = self._get_vqs_state(params_coeffs)
        
        # 2. Get the exact solution (ExactRTE or ExactITE)
        psi_exact = self.exact_solver.get_exact_state(t)
        
        # 3. Compute the fidelity |<exact|vqs>|^2
        overlap = jnp.vdot(psi_exact, psi_vqs)
        return jnp.abs(overlap) ** 2

    def compute_batched(self, params_all, time_points, batch_size=100):
        """Compute fidelities along the entire evolution trajectory in batches"""
        # params_all should have shape (T, Total_params)
        T = params_all.shape[0]
        fidelities = []
        n_batches = math.ceil(T / batch_size)

        for i in tqdm(range(0, T, batch_size), total=n_batches, desc="Computing Fidelity"):
            p_batch = params_all[i : i + batch_size]
            t_batch = time_points[i : i + batch_size]
            
            f = self._vmapped_fidelity(p_batch, t_batch)
            fidelities.append(f)

        return jnp.concatenate(fidelities, axis=0)



class MultiStateEnergyCalculator:
    """Class for comparing the energies of a variational superposition (VQS) and exact imaginary-time evolution (ITE)"""
    def __init__(self, circuit, exact_solver, hamiltonian_matrix, Ns):
        """
        Args:
            circuit: A single ansatz circuit function
            exact_solver: An instance of ExactITE
            hamiltonian_matrix: The system Hamiltonian matrix (H_mat)
            Ns: Number of component states in the superposition
        """
        self.circuit = circuit
        self.exact_solver = exact_solver
        self.H_mat = hamiltonian_matrix
        self.Ns = Ns
        
        # Predefine a batched circuit function to compute Ns component states in parallel
        self.circuits_fn = jax.jit(jax.vmap(self.circuit, in_axes=(0,)))
        
        # Set up the batched energy calculation function for each time t and parameter vector params_coeffs along the trajectory
        self._vmapped_energy = jax.vmap(self.calc_energy, in_axes=(0, 0))

    @partial(jax.jit, static_argnums=(0,))
    def _get_expectation(self, psi):
        """Compute the normalized energy expectation value: <psi|H|psi> / <psi|psi>"""
        # Compute <psi|H|psi>
        expectation_val = jnp.vdot(psi, self.H_mat @ psi).real
        # Compute <psi|psi>
        norm_sq = jnp.vdot(psi, psi).real
        return expectation_val / norm_sq

    @partial(jax.jit, static_argnums=(0,))
    def _get_vqs_state(self, params_coeffs):
        """Internal method that constructs a superposition from the parameters; normalization is handled by the energy calculation"""
        # 1. Parameter slices: [theta_1...theta_n, a_1...a_Ns, b_1...b_Ns]
        theta = params_coeffs[:-2*self.Ns].reshape(self.Ns, -1)
        coeffs_a = params_coeffs[-2*self.Ns : -self.Ns]
        coeffs_b = params_coeffs[-self.Ns :]
        
        # 2. Compute component states (Ns, dim)
        states = self.circuits_fn(theta)
        
        # 3. Form the linear superposition with c_i = a_i * exp(i * b_i)
        coeffs_c = coeffs_a * jnp.exp(1j * coeffs_b)
        psi = coeffs_c @ states
        return psi  # unnormalized

    @partial(jax.jit, static_argnums=(0,))
    def calc_energy(self, params_coeffs, t):
        """Compute the VQS and exact ITE energies at a given parameter vector and time"""
        # 1. Construct the variational superposition and compute its energy
        psi_vqs = self._get_vqs_state(params_coeffs)
        E_vqs = self._get_expectation(psi_vqs)
        
        # 2. Get the exact ITE state and compute its energy
        psi_exact = self.exact_solver.get_exact_state(t)
        E_ite = self._get_expectation(psi_exact)
        
        return E_vqs, E_ite

    def compute_batched(self, params_all, time_points, batch_size=100):
        """Compute energies along the evolution trajectory in batches"""
        # params_all shape: (T, Total_params)
        T = params_all.shape[0]
        e_vqs_list = []
        e_ite_list = []
        n_batches = math.ceil(T / batch_size)

        for i in tqdm(range(0, T, batch_size), total=n_batches, desc="Computing Energy"):
            p_batch = params_all[i : i + batch_size]
            t_batch = time_points[i : i + batch_size]
            
            # Call vmap to return two one-dimensional arrays of length batch_size
            e_vqs, e_ite = self._vmapped_energy(p_batch, t_batch)
            
            e_vqs_list.append(e_vqs)
            e_ite_list.append(e_ite)

        # Concatenate the results from all batches
        return jnp.concatenate(e_vqs_list, axis=0), jnp.concatenate(e_ite_list, axis=0)


class MultiStateOccupationCalculator:
    """Compute all spin-orbital occupations for the NOVQS variational state and the exact state."""

    def __init__(self, circuit, exact_solver, Ns, Nq):
        """
        Args:
            circuit: A single ansatz circuit that takes a parameter set and returns a state vector
            exact_solver: An instance of ExactRTE or ExactITE
            Ns: Number of component states in NOVQS
            Nq: Number of qubits, equal to the number of spin orbitals
        """
        self.circuit = circuit
        self.exact_solver = exact_solver
        self.Ns = Ns
        self.Nq = Nq

        # Take Ns sets of circuit parameters and return Ns component states together
        self.circuits_fn = jax.jit(jax.vmap(self.circuit, in_axes=0))

        # occupation_table[k, p] is the occupation of orbital p in computational basis state k
        self.occupation_table = self._build_occupation_table()

        # Compute NOVQS and exact occupations in batches over time
        self._vmapped_occupations = jax.jit(
            jax.vmap(self.calc_occupations, in_axes=(0, 0))
        )

    def _build_occupation_table(self):
        """Build the spin-orbital occupation table for all computational basis states."""
        dim = 2**self.Nq
        basis_indices = jnp.arange(dim, dtype=jnp.uint32)
        bit_positions = jnp.arange(self.Nq - 1, -1, -1, dtype=jnp.uint32)

        occupation_table = (
            (basis_indices[:, None] >> bit_positions[None, :]) & 1
        )

        return occupation_table

    @partial(jax.jit, static_argnums=(0,))
    def _get_vqs_state(self, params_coeffs):
        """Construct the normalized NOVQS superposition from the parameters."""
        theta = params_coeffs[:-2 * self.Ns].reshape(self.Ns, -1)
        coeffs_a = params_coeffs[-2 * self.Ns:-self.Ns]
        coeffs_b = params_coeffs[-self.Ns:]

        states = self.circuits_fn(theta)
        coeffs_c = coeffs_a * jnp.exp(1j * coeffs_b)
        psi_vqs = coeffs_c @ states

        return psi_vqs / jnp.linalg.norm(psi_vqs)

    @partial(jax.jit, static_argnums=(0,))
    def _occupations_from_state(self, state):
        """Compute all spin-orbital occupations from a state vector."""
        probabilities = jnp.abs(state) ** 2
        probabilities = probabilities / jnp.sum(probabilities)
        occupation_table = self.occupation_table.astype(probabilities.dtype)

        return probabilities @ occupation_table

    @partial(jax.jit, static_argnums=(0,))
    def calc_vqs_occupations(self, params_coeffs):
        """Compute NOVQS spin-orbital occupations at a single time point."""
        psi_vqs = self._get_vqs_state(params_coeffs)
        return self._occupations_from_state(psi_vqs)

    @partial(jax.jit, static_argnums=(0,))
    def calc_exact_occupations(self, t):
        """Compute exact spin-orbital occupations at a single time point."""
        psi_exact = self.exact_solver.get_exact_state(t)
        return self._occupations_from_state(psi_exact)

    @partial(jax.jit, static_argnums=(0,))
    def calc_occupations(self, params_coeffs, t):
        """Compute both NOVQS and exact occupations at a single time point."""
        occupations_vqs = self.calc_vqs_occupations(params_coeffs)
        occupations_exact = self.calc_exact_occupations(t)

        return occupations_vqs, occupations_exact

    def compute_batched(self, params_all, time_points, batch_size=100):
        """Compute NOVQS and exact occupations at all time points in batches."""
        n_times = params_all.shape[0]

        vqs_batches = []
        exact_batches = []
        n_batches = math.ceil(n_times / batch_size)

        for i in tqdm(
            range(0, n_times, batch_size),
            total=n_batches,
            desc="Computing occupations",
        ):
            params_batch = params_all[i:i + batch_size]
            time_batch = time_points[i:i + batch_size]

            occ_vqs, occ_exact = self._vmapped_occupations(params_batch, time_batch)

            vqs_batches.append(occ_vqs)
            exact_batches.append(occ_exact)

        occupations_vqs = jnp.concatenate(vqs_batches, axis=0)
        occupations_exact = jnp.concatenate(exact_batches, axis=0)

        return occupations_vqs, occupations_exact        