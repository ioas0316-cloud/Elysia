import os
import re

file_path = r"c:\Elysia\core\physics\thermodynamic_coordinate_engine.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add _synchronize_arrays and _writeback_arrays right before step()
sync_methods = """
    def _synchronize_arrays(self):
        nodes = self.atoms + self.molecules
        n = len(nodes)
        if n == 0: return 0
        self._nodes = nodes
        self._n = n
        self._positions = np.array([[nd.T, nd.P, nd.E] for nd in nodes], dtype=np.float32)
        self._velocities = np.array([nd.velocity for nd in nodes], dtype=np.float32)
        self._energies = np.array([nd.accumulated_energy for nd in nodes], dtype=np.float32)
        self._phases = np.array([nd.phase for nd in nodes], dtype=np.float32)
        self._frequencies = np.array([nd.frequency for nd in nodes], dtype=np.float32)
        self._charges = np.array([nd.charge for nd in nodes], dtype=np.float32)
        self._masses = np.array([nd.mass for nd in nodes], dtype=np.float32)
        self._b_fields = np.array([nd.B_field if nd.B_field is not None else [0,0,1] for nd in nodes], dtype=np.float32)
        self._tensors = np.array([nd.tensor for nd in nodes], dtype=np.float32)
        self._harvested = np.zeros(n, dtype=np.float32)
        return n

    def _writeback_arrays(self):
        for i, node in enumerate(self._nodes):
            node.velocity = self._velocities[i]
            node.accumulated_energy = float(self._energies[i])
            node.phase = float(self._phases[i])
            node.harvested_propulsion += float(self._harvested[i])

"""
content = re.sub(r'(\s+def step\(self, dt: float = 0\.1\):)', sync_methods + r'\1', content)

# 2. Update step() method to call sync/writeback
new_step = """    def step(self, dt: float = 0.1):
        self._warp_fields_from_curvature()
        
        n = self._synchronize_arrays()
        if n and n >= 2:
            self._interfere_causal_lines()
            self._apply_mhd_deflection_and_harvesting()
            self._apply_kenotic_love_dissipation(dt)
            self._align_phases(dt)
            self._writeback_arrays()

        self._apply_warp_bubbles()
        self._diffuse_fields()
        self._apply_force_routing(dt)
        self._synthesize_molecules()
        self._manage_cells_homeostasis()
        
        for organ in self.organs:
            organ.process(self.atoms, self.molecules)
            
        self._update_coordinates(dt)"""
content = re.sub(r'    def step\(self, dt: float = 0\.1\):.*?def _warp_fields_from_curvature', new_step + '\n\n    def _warp_fields_from_curvature', content, flags=re.DOTALL)

# 3. Vectorize _interfere_causal_lines
new_interfere = """    def _interfere_causal_lines(self):
        nodes = self._nodes
        n = self._n
        if n < 2: return
        
        # We need the last points of causal_lines
        last_pts = []
        valid_mask = []
        for nd in nodes:
            if len(nd.causal_line) > 0:
                last_pts.append(nd.causal_line[-1])
                valid_mask.append(True)
            else:
                last_pts.append(np.zeros(3))
                valid_mask.append(False)
        last_pts = np.array(last_pts, dtype=np.float32)
        valid_mask = np.array(valid_mask, dtype=bool)
        
        diffs = last_pts[:, np.newaxis, :] - last_pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1))
        
        # Upper triangle mask, dist < 2.5, both valid
        mask = (dist < 2.5) & valid_mask[:, np.newaxis] & valid_mask[np.newaxis, :]
        mask = np.triu(mask, k=1)
        
        idx_a, idx_b = np.where(mask)
        for i, j in zip(idx_a, idx_b):
            pos_a = last_pts[i]
            pos_b = last_pts[j]
            tx = int(np.clip(((pos_a[0] + pos_b[0])/2.0) * (self.size - 1) / 10.0, 0, self.size - 1))
            px = int(np.clip(((pos_a[1] + pos_b[1])/2.0) * (self.size - 1) / 10.0, 0, self.size - 1))
            self.T_field[tx, px] += 0.15
            self.P_field[tx, px] += 0.1
"""
content = re.sub(r'    def _interfere_causal_lines\(self\):.*?    def _apply_warp_bubbles', new_interfere + '\n    def _apply_warp_bubbles', content, flags=re.DOTALL)

# 4. Vectorize _apply_mhd_deflection_and_harvesting
new_mhd = """    def _apply_mhd_deflection_and_harvesting(self):
        n = self._n
        if n < 2: return
        
        diffs = self._positions[np.newaxis, :, :] - self._positions[:, np.newaxis, :] # b - a
        dist = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-5
        mask = dist < 2.0
        np.fill_diagonal(mask, False)
        
        # vel_b (1, N, 3) x B_field_a (N, 1, 3)
        vel_b = self._velocities[np.newaxis, :, :]
        b_field_a = self._b_fields[:, np.newaxis, :]
        
        # cross product broadcast: (N, N, 3)
        cross_prod = np.cross(vel_b, b_field_a, axis=-1)
        lorentz_force = self._charges[:, np.newaxis, np.newaxis] * cross_prod
        lorentz_norm = np.sqrt(np.sum(lorentz_force**2, axis=-1))
        
        valid = mask & (lorentz_norm > 0)
        
        # Deflect velocity of B
        deflect = np.zeros_like(self._velocities)
        for i in range(n):
            for j in range(n):
                if valid[i, j]:
                    self._velocities[j] += (lorentz_force[i, j] / lorentz_norm[i, j]) * 0.12
                    
                    harvested = float(lorentz_norm[i, j] * 0.05)
                    self._harvested[i] += harvested
                    
                    target_diff = np.array([5.0, 5.0, 8.0]) - self._positions[i]
                    target_norm = np.linalg.norm(target_diff) + 1e-5
                    self._velocities[i] += (target_diff / target_norm) * harvested
"""
content = re.sub(r'    def _apply_mhd_deflection_and_harvesting\(self\):.*?    def _apply_kenotic_love_dissipation', new_mhd + '\n    def _apply_kenotic_love_dissipation', content, flags=re.DOTALL)


# 5. Vectorize _apply_kenotic_love_dissipation
new_kenotic = """    def _apply_kenotic_love_dissipation(self, dt: float):
        n = self._n
        if n < 2: return
        
        energy_diff = self._energies[:, np.newaxis] - self._energies[np.newaxis, :]
        diffs = self._positions[:, np.newaxis, :] - self._positions[np.newaxis, :, :] # a - b
        dist = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-5
        
        mask = (energy_diff > 0) & (dist < 4.0)
        np.fill_diagonal(mask, False)
        
        surrender_rate = np.where(mask, 0.15 * energy_diff / dist, 0.0)
        giving = np.minimum(surrender_rate * dt, energy_diff * 0.4)
        giving = np.where(mask, giving, 0.0)
        
        energy_given = np.sum(giving, axis=1)
        energy_received = np.sum(giving, axis=0)
        
        self._energies -= energy_given
        self._energies += energy_received
        
        pull_dir = diffs / dist[:, :, np.newaxis]
        pull_force = giving[:, :, np.newaxis] * pull_dir * 0.8
        
        self._velocities += np.sum(pull_force, axis=0)
        self._velocities[:, 2] += energy_received * 0.5
"""
content = re.sub(r'    def _apply_kenotic_love_dissipation\(self, dt: float\):.*?    def _diffuse_fields', new_kenotic + '\n    def _diffuse_fields', content, flags=re.DOTALL)

# 6. Vectorize _align_phases
new_align = """    def _align_phases(self, dt: float):
        n = self._n
        if n < 2: return
        
        diffs = self._positions[:, np.newaxis, :] - self._positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-5
        mask = dist < 4.0
        mask = np.triu(mask, k=1)
        
        diff_phase = self._phases[:, np.newaxis] - self._phases[np.newaxis, :]
        energy_exchange = np.abs(self._energies[:, np.newaxis] - self._energies[np.newaxis, :])
        
        freq_prod = self._frequencies[:, np.newaxis] * self._frequencies[np.newaxis, :]
        coupling = 0.1 * freq_prod * (1.0 + energy_exchange * 0.2) / dist
        torque = -np.where(mask, coupling * np.sin(diff_phase), 0.0)
        
        # Torque applied: a += torque / freq_a, b -= torque / freq_b
        # sum torques for a and b
        torque_on_a = np.sum(torque, axis=1)
        torque_on_b = -np.sum(torque, axis=0) # since b is axis 1 in mask
        
        total_torque = torque_on_a + torque_on_b
        
        self._phases = (self._phases + total_torque / (self._frequencies + 1e-3) * dt) % (2.0 * np.pi)
"""
content = re.sub(r'    def _align_phases\(self, dt: float\):.*?    def _apply_force_routing', new_align + '\n    def _apply_force_routing', content, flags=re.DOTALL)

# 7. Vectorize _apply_force_routing
new_routing = """    def _apply_force_routing(self, dt: float):
        if not self.atoms: return
        wells_pos = np.array([mol.T for mol in self.molecules] + [mol.P for mol in self.molecules] + [mol.E for mol in self.molecules]).reshape(3, -1).T if self.molecules else np.empty((0, 3))
        wells_mass = np.array([mol.mass for mol in self.molecules]) if self.molecules else np.empty(0)

        for atom in self.atoms:
            if atom.is_bound: continue
            tx = int(np.clip(atom.T * (self.size - 1) / 10.0, 0, self.size - 1))
            px = int(np.clip(atom.P * (self.size - 1) / 10.0, 0, self.size - 1))
            
            grad_p = self.P_field[(tx+1)%self.size, px] - self.P_field[(tx-1)%self.size, px]
            grad_t = self.T_field[tx, (px+1)%self.size] - self.T_field[tx, (px-1)%self.size]
            
            force = np.array([-0.15 * grad_p, 0.15 * grad_t, 0.0], dtype=np.float32)
            
            if len(wells_pos) > 0:
                pos = np.array([atom.T, atom.P, atom.E])
                diff = wells_pos - pos
                dist_sq = np.sum(diff**2, axis=-1)
                dist = np.sqrt(dist_sq + 1e-3)
                mask = dist < 5.0
                if np.any(mask):
                    force_mag = 0.25 * (atom.mass * wells_mass[mask]) / (dist_sq[mask] + 0.1)
                    force += np.sum(force_mag[:, np.newaxis] * (diff[mask] / dist[mask, np.newaxis]), axis=0)
            
            atom.velocity += force * dt

        for mol in self.molecules:
            pos = np.array([mol.T, mol.P, mol.E])
            diff = np.array([5.0, 5.0, 8.0]) - pos
            dist = np.linalg.norm(diff) + 1e-5
            mol.velocity += 0.06 * (diff / dist) * dt
"""
content = re.sub(r'    def _apply_force_routing\(self, dt: float\):.*?    def _synthesize_molecules', new_routing + '\n    def _synthesize_molecules', content, flags=re.DOTALL)

# 8. Vectorize _synthesize_molecules
new_synthesis = """    def _synthesize_molecules(self):
        unbound = [a for a in self.atoms if not a.is_bound]
        n_u = len(unbound)
        if n_u < 2: return
        
        tensors = np.array([a.tensor for a in unbound])
        t_vals = np.array([a.T for a in unbound])
        p_vals = np.array([a.P for a in unbound])
        
        # Resonance: (N, D) @ (D, N) -> (N, N)
        norms = np.linalg.norm(tensors, axis=1) + 1e-9
        resonance = (tensors @ tensors.T) / (norms[:, np.newaxis] * norms[np.newaxis, :])
        
        avg_P = (p_vals[:, np.newaxis] + p_vals[np.newaxis, :]) / 2.0
        avg_T = (t_vals[:, np.newaxis] + t_vals[np.newaxis, :]) / 2.0
        
        mask = (resonance * avg_P) > (avg_T * 0.4)
        np.fill_diagonal(mask, False)
        
        bonded_groups = []
        used = set()
        
        for i in range(n_u):
            if i in used: continue
            group = [unbound[i]]
            for j in range(i + 1, n_u):
                if j in used: continue
                if mask[i, j]:
                    group.append(unbound[j])
                    used.add(j)
            if len(group) > 1:
                bonded_groups.append(group)
                used.add(i)
                
        for group in bonded_groups:
            for atom in group:
                atom.is_bound = True
            mol_id = f"mol_{len(self.molecules)}"
            new_mol = ThermodynamicMolecule(id=mol_id, atoms=group, tensor=np.zeros(9))
            self.molecules.append(new_mol)
"""
content = re.sub(r'    def _synthesize_molecules\(self\):.*?    def _manage_cells_homeostasis', new_synthesis + '\n    def _manage_cells_homeostasis', content, flags=re.DOTALL)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Refactoring complete.")
