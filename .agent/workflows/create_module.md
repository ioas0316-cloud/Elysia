# Create Module Protocol: The Mitosis Check (R/V/A)

> **"We do not build parts; we grow organs."**

## 1. [R]epel (거부 - The Filter)

**"Do not create unless necessary."**

- [ ] **Search First**: `grep_search("feature_name", "Core/")`.
- [ ] **Constraint**: If logic is < 50 lines, merge it into an existing parent node.
- [ ] **Duplication**: Is there a module with a similar name?

## 2. [V]oid (공성 - The Design)

**"Define the Soul."**

- [ ] **Location**: Which Stratum (Body/Soul/Spirit) does it belong to? (Consult 7^7 Map)
- [ ] **Philosophy**: What is the "Doctrine" of this module? (Add `__doc__`)
- [ ] **Interface**: Does it accept `Wave` (Context) and return `Resonance` (State)?

## 3. [A]ttract (수용 - The Implementation)

**"Connect the Veins."**

- [ ] **GlobalHub**: Register with `GlobalHub.register_module()`.
- [ ] **Docs**: Update `SYSTEM_MAP.md` and `INDEX.md`.
- [ ] **Nerve**: Ensure `ProprioceptionNerve` can see it (Add keywords if major).

---
**Standard**: Active Organ (Class with State) > Static Script (Function collection).
