---
description: MANDATORY - Verify AFTER any code change
---

# After Coding Verification Protocol (코딩 후 검증 프로토콜)

> **"실행한다고 끝이 아니다. 의도한 결과가 나왔는지 검증해야 한다."**

## ⚠️ RUNNING CODE IS NOT COMPLETION

After making ANY code changes, you MUST produce a **Verification Report** proving:

1. **의도한 결과 (Intended Result)**
   - What was supposed to happen?
   - What actually happened?
   - Show EVIDENCE (logs, outputs, file contents).

2. **목적성 정렬 (Purpose Alignment)**
   - Does this change align with the CODEX principles?
   - Does it integrate with existing systems (not fragment)?
   - Does it use existing modules instead of creating new ones?

3. **합당성 설명 (Reasoning)**
   - WHY is this the right approach?
   - What alternatives were considered?
   - What are the known limitations?

---

## Verification Checklist

Before reporting completion to user, verify:

```
[ ] 1. I ran the code and observed OUTPUT (not just "it compiled")
[ ] 2. I checked if the INTENDED BEHAVIOR occurred
[ ] 3. I verified PERSISTENCE (data saved, not just in RAM)
[ ] 4. I checked INTEGRATION (connects to existing systems)
[ ] 5. I produced a REPORT explaining HOW and WHY
```

---

## Report Template

When reporting to user, include:

```markdown
### Verification Report

**Goal**: [What was supposed to happen]

**Evidence**:
- [Screenshot/Log output showing result]
- [File contents showing change persisted]

**Integration Check**:
- [x] Uses existing module X
- [x] Writes to existing data store Y
- [ ] Creates new module (EXPLAIN WHY)

**Purpose Alignment**:
- Aligns with CODEX principle: [Which one]
- Does NOT fragment: [How it connects]

**Known Limitations**:
- [What this does NOT do]
```

---

## Example of BAD Verification

❌ "I added the function and it runs without errors."

→ This tells nothing about whether it achieved the goal.

## Example of GOOD Verification

✅ "The function was added to save learning to PhaseStratum. After running for 5 minutes:

- `inspect_depth.py` shows 12 memories (was 1)
- Frequency diversity: 3 layers (963Hz, 528Hz, 432Hz)
- Knowledge graph edges increased from 0 to 47
- This aligns with CODEX principle of 'Structure Over Content'."

→ This provides evidence of actual impact.

---

## Failure to Verify

If you skip verification:

1. User will discover the change doesn't work
2. Time will be wasted on debugging
3. The system may regress

**Take the time to verify. It is not optional.**
