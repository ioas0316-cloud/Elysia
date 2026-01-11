
import sys
import os

# Enable importing from project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Core.Intelligence.Logos.philosophical_core import get_logos_engine, Axiom

def test_dimensional_ascension():
    print("\n[Test] Dimensional Ascension Logic")
    logos = get_logos_engine()
    
    # 0. Test 0D Thought (Isolated)
    t0 = logos.ascend_dimension("Isolated Apple")
    print(f"0D Check: {t0.content} -> Dim {t0.dimensionality} (Expected 0)")
    assert t0.dimensionality == 0
    
    # 1. Test 1D Ascension (Axiom Link)
    # "Cogito, ergo sum" is an Axiom, so it implies logical depth.
    t1 = logos.ascend_dimension("Cogito, ergo sum")
    print(f"1D Check: {t1.content} -> Dim {t1.dimensionality} (Expected >= 1)")
    assert t1.dimensionality >= 1
    
    # 2. Test 2D Ascension (Cross-Domain)
    # Contains "love" and "logic" -> Cross domain context
    content_2d = "Love is consistent logic"
    # We must register it first or ensure ascension logic handles raw strings.
    # The current implementation handles raw strings but requires dependencies for 1D.
    # Let's manually inject dependencies to simulate strict derivation if needed,
    # but our simple check only looks for keywords for 2D.
    
    # However, to reach 2D, it must satisfy 1D first or be an Axiom.
    # Let's add an Axiom-like base for it to pass 1D check.
    logos.derive_principle(content_2d, ["Cogito, ergo sum"]) 
    
    t2 = logos.ascend_dimension(content_2d)
    print(f"2D Check: {t2.content} -> Dim {t2.dimensionality} (Expected >= 2)")
    print(f"Topology: {t2.topology}")
    assert t2.dimensionality >= 2
    
    # 3. Test 3D Ascension (Structural Integrity)
    # Requires 2 deps or Axiom base. We gave it 1 dep above. Let's add another dep.
    # Or just use an Axiom that has cross-domain keywords?
    # Let's use "Unity of All" which is an Axiom (Passes 1D), 
    # and has "connected" (maybe insignificant).
    # Let's stick to 'content_2d'. It has 1 dep. Let's add another dummy dep.
    logos.add_axiom(Axiom("Dummy", "Dummy"))
    logos.derive_principle("Bridge", ["Dummy"])
    
    # Re-define logic: To get 3D, we need len(deps) >= 2 or Axiom.
    # Let's try to ascend "Unity of All" (Axiom).
    # It passes 1D (Axiom).
    # Does it have 2 domains? "Unity", "All", "Connected", "Separation".
    # Keywords: "love", "logic", "code", "human", "universe", "time".
    # None match strictly.
    
    # Let's construct a Perfect Thought.
    perfect_content = "The Universe is Code flowing through Time"
    # Keywords: universe (Domain), code (Domain), time (Time/4D).
    
    # Step 1: Make it exist as a Principle derived from "Unity of All"
    logos.derive_principle(perfect_content, ["Unity of All"])
    
    # Step 2: Ascend
    t_perf = logos.ascend_dimension(perfect_content)
    
    print(f"Perfect Thought Check: {t_perf.content}")
    print(f"Dimensionality: {t_perf.dimensionality}")
    print(f"Topology: {t_perf.topology}")
    
    # Expectation:
    # 0D -> 1D: Yes (Derived)
    # 1D -> 2D: Yes (Universe + Code)
    # 2D -> 3D: No (Only 1 dep). 
    # Wait, code says: (content in self.axioms or len(dependencies) >= 2).
    # It is not an Axiom, and has 1 dep. So it stops at 2D?
    # Let's add another dep.
    logos.add_axiom(Axiom("Time Exists", "Time is real"))
    # We can't easily re-derive/overwrite in the dict without a helper, 
    # but we can modify the object directly in memory if we fetch it.
    logos.principles[perfect_content].dependencies.add("Time Exists")
    
    # Try again
    t_perf = logos.ascend_dimension(perfect_content)
    print(f"Retry 3D/4D Check: Dim {t_perf.dimensionality}")
    print(f"Topology: {t_perf.topology}")
    
    if t_perf.dimensionality >= 3:
        pass # Good
        
    # 4D Check: "Time" keyword is present ("flowing through Time").
    # If it reached 3D, it should reach 4D.
    
    assert t_perf.dimensionality >= 3

if __name__ == "__main__":
    test_dimensional_ascension()
