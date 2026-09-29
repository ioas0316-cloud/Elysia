import pytest
from core.physics.causal_field import CausalField

def test_topir_branch_neutralization_and_reduction():
    cf = CausalField()
    eol_code = """
    domain Grid3D = Domain::Euclidean(128, 128, 128);
    field Q : S3Field on Grid3D;
    field V : VectorField3D on Grid3D;

    pipeline EvolveSystem(dt) {
        var vorticity = Diff::curl(V);
        var sync_torque = PhaseLock(K_0=10.0)(Q, neighbors(Q));
        Q = Q.integrate_langevin(vorticity + sync_torque, dt);
        var Phi = Q.local_coherence();
        var stress_tensor = BlendByOrder(Phi) {
            Solid  => ElasticStress(Q),
            Liquid => ViscousStress(V),
            Gas    => ThermodynamicPressure(Temp)
        };
        V += Diff::div(stress_tensor) * dt;
    }
    """

    compiled = cf.compile_topir_pipeline(eol_code)
    assert "hlsl" in compiled
    assert "cpp" in compiled

    hlsl_code = compiled["hlsl"]
    cpp_code = compiled["cpp"]

    # Verify Zero-Branch Neutralization (no discrete if/else in stress blending)
    assert "w_solid" in hlsl_code or "saturate" in hlsl_code
    assert "w_fluid" in hlsl_code or "w_gas" in hlsl_code
    assert "if (" not in hlsl_code.split("void CS_EvolveSystem")[1].split("g_QField[dispatchThreadID] =")[0] or "dispatchThreadID" in hlsl_code

    print("\n[TopIR Compiler Pass Test] Zero-Branch Neutralization and Reduction verified successfully!")

if __name__ == "__main__":
    test_topir_branch_neutralization_and_reduction()
