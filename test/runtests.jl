import LatticeSurgeryCCZStateExample as LS
using Test

function basic_circuit_components()
    tag = "mem"

    test_gates = vcat(LS.zzzz_meas(tag, (1, 3), 1),
        LS.down_xx_meas(tag, (1, 5), 1),
        LS.up_xx_meas(tag, (1, 1), 1),
        LS.zzzz_meas(tag, (1, 3), 2),
        LS.down_xx_meas(tag, (1, 5), 2),
        LS.up_xx_meas(tag, (1, 1), 2))

    test_stabs = [LS.zzzz_stab((1, 3)),
                LS.down_xx_stab((1, 5)),
                LS.up_xx_stab((1, 1))]

    test_layers = [LS.Layer(test_gates, test_stabs)]

    test_fault = LS.Fault(LS.identity_pauli(LS.nq), 1, 0)

    test_log_ops = ([LS.z_on_set([(0, 4), (2, 4)])],
                        [LS.x_on_set([(0, 2), (0, 4)])])

    function test_log_meas_res(comp_state::LS.ComputationalState)
        Dict{String, UInt8}()
    end

    function test_postselect(comp_state::LS.ComputationalState, log_results)
        any(val == 0x01 for val in values(comp_state.meas_output))
    end

    function test_log_corrections(comp_state, log_results)
        comp_state
    end

    (test_layers, test_fault, test_stabs, test_log_ops,
            test_log_meas_res, test_postselect, test_log_corrections)
end

function basic_circuit()
    LS.Circuit(basic_circuit_components()...)
end

@testset "Pauli convenience functions" begin
    ps_5_99 = LS.paulis_on(LS.nq, [5, 99])
    ps_5 = LS.paulis_on(LS.nq, 5)
    ps_99 = LS.paulis_on(LS.nq, 99)
    
    @test length(ps_5) == 3
    @test length(ps_99) == 3
    
    q_5 = LS.all_qubits[5]
    xyz_5 = [LS.x_on(q_5), LS.z_on(q_5),
                LS.phase_free_prod([LS.x_on(q_5), LS.z_on(q_5)])]
    
    q_99 = LS.all_qubits[99]
    xyz_99 = [LS.x_on(q_99), LS.z_on(q_99),
                LS.phase_free_prod([LS.x_on(q_99), LS.z_on(q_99)])]
    
    for pauli in xyz_5
        @test pauli in ps_5
        @test pauli in ps_5_99 
    end 

    for pauli in xyz_99
        @test pauli in ps_99
        @test pauli in ps_5_99 
    end

    @test length(ps_5_99) == 15
    for p_5 in ps_5 
        for p_99 in ps_99 
            @test p_5 * p_99 in ps_5_99
        end
    end
end

@testset "Supports, qubit sets and weights" begin
    @test LS.weight(LS.identity_pauli(10)) == 0
    
    qs = [(0, 0), (1, 1)]
    dxs = map(q -> LS.addrs[q], qs)
    x_2 = LS.x_on_set(qs)
    @test Set(LS.support(LS.x_on_set(qs))) == Set(dxs)
    @test Set(LS.support(LS.z_on_set(qs))) == Set(dxs)
    @test Set(LS.qubits(LS.QC.sCNOT(1, 2))) == Set([1, 2])
    @test Set(LS.qubits(LS.QC.sPhase(12))) == Set([12])

    xyz = LS.xz_on_set(LS.all_qubits[[1, 2]], LS.all_qubits[[2, 3]])
    @test LS.weight(xyz) == 3
    @test xyz == LS.x_on_set(LS.all_qubits[[1, 2]]) * LS.z_on_set(LS.all_qubits[[2, 3]])
end

@testset "Preparation Basics" begin
    test_x = LS.x_on((3, 3))
    test_z = LS.z_on((3, 3))
    other_test_x = LS.x_on((3, 3))
    other_test_z = LS.z_on((5, 3))

    @test LS.Preparation(test_x) == LS.Preparation(other_test_x)
    @test ~(LS.Preparation(test_z) == LS.Preparation(other_test_z))
    @test ~(LS.Preparation(test_z) == LS.Preparation(test_x))
    @test ~(LS.Preparation(test_x) == LS.Preparation(test_z))
end

@testset "Fault basics" begin
    pauli = LS.x_on_set([(1, 1), (1, 3)]) * LS.z_on_set([(1, 3), (3, 3)])
    test_fault = LS.Fault(pauli, 2, 7)
    @test test_fault.pauli == pauli 
    @test test_fault.layer_dx == 2
    @test test_fault.gate_dx == 7 
end

@testset "Layer basics" begin
    gates = [LS.QC.sCNOT(1, 2), LS.QC.sCNOT(3, 4)]
    stabilisers = [LS.z_on_set([(0, 0), (1, 1)]), LS.x_on_set([(0, 0), (1, 1)])]

    first_layer = LS.Layer(gates, stabilisers)
    @test LS.gatelist(first_layer) == first_layer.gatelist
    @test LS.gatelist(first_layer) == gates
    @test Set(LS.qubits(first_layer)) == Set([1, 2, 3, 4])
    @test LS.stabs(first_layer) == stabilisers

    more_gates = [LS.QC.sCNOT(5, 6), LS.QC.sCNOT(7, 8)]
    
    second_layer = LS.Layer(more_gates, stabilisers)
    @test Set(LS.qubits([first_layer, second_layer])) == Set(collect(1:8))
end

@testset "Circuit Basics" begin
    components = basic_circuit_components()
    test_layers = components[1]
    test_fault = components[2]
    test_stabs = components[3]
    test_log_ops = components[4]
    test_log_meas_res = components[5]
    test_postselect = components[6]
    test_log_corrections = components[7]

    test_circuit = basic_circuit()

    qs = [(1, 1), (1, 3), (1, 5), (0, 2), (0, 4), (2, 2), (2, 4)]
    dxs = Set(map(q -> LS.addrs[q], qs))
    
    @test LS.layers(test_circuit) == test_layers
    @test LS.fault(test_circuit) == test_fault
    @test Set(LS.qubits(test_circuit)) == dxs

    new_fault = LS.Fault(LS.x_on((0, 4)), 1, 10)
    test_circuit = LS.add_fault(test_circuit, new_fault)
    
    @test LS.layers(test_circuit) == test_layers
    @test Set(LS.qubits(test_circuit)) == dxs
    @test LS.fault(test_circuit) == new_fault

    # circuit_after_fault
end

@testset "ComputationalState Basics" begin
    test_circuit = basic_circuit()
    comp_state = LS.ComputationalState(test_circuit)
    pauli, meas_output = comp_state.pauli, comp_state.meas_output
    @test length(keys(meas_output)) == 6
    @test length(values(meas_output)) == 6
    for val in values(meas_output)
        @test val == 0x00
    end
    @test LS.weight(pauli) == 0
end

@testset "Applying Preparations to ComputationalStates" begin
    pauli_on_prep_qubit = LS.x_on((3, 3))
    pauli_off_prep_qubit = LS.z_on((5, 5))
    test_meas_res = Dict("a" => 0x00, "b" => 0x01)
    test_prep = LS.Preparation(pauli_on_prep_qubit)

    test_state = LS.ComputationalState(pauli_on_prep_qubit, test_meas_res)
    new_state = LS.apply(test_prep, test_state)
    @test new_state.meas_output == test_state.meas_output
    @test LS.weight(new_state.pauli) == 0
    @test test_state.pauli == pauli_on_prep_qubit

    test_state = LS.ComputationalState(pauli_off_prep_qubit, test_meas_res)
    new_state = LS.apply(test_prep, test_state)
    @test new_state.meas_output == test_state.meas_output
    @test test_state.pauli == pauli_off_prep_qubit
end

@testset "NotNot Basics" begin
    test_notnot = LS.NotNot(1, 2)
    @test Set(LS.qubits(test_notnot)) == Set([1, 2])
    xi = LS.QC.P"XI"
    ix = LS.QC.P"IX"
    zi = LS.QC.P"ZI"
    iz = LS.QC.P"IZ"
    xz = LS.QC.P"XZ"
    zx = LS.QC.P"ZX"

    @test LS.apply_gate_to_pauli(test_notnot, xi) == xi
    @test LS.apply_gate_to_pauli(test_notnot, ix) == ix
    @test LS.apply_gate_to_pauli(test_notnot, zi) == zx
    @test LS.apply_gate_to_pauli(test_notnot, iz) == xz
end

@testset "Applying Gates to ComputationalStates" begin
    test_meas_res = Dict("a" => 0x00, "b" => 0x01)
    test_pauli = LS.z_on((5, 5))
    ctrl, targ = LS.addrs[(6, 6)], LS.addrs[(5, 5)]
    test_cnot = LS.QC.sCNOT(ctrl, targ)
    test_notnot = LS.NotNot(ctrl, targ)

    test_state = LS.ComputationalState(test_pauli, test_meas_res)
    new_state = LS.apply(test_cnot, test_state)
    
    @test test_state.pauli == test_pauli
    @test new_state.pauli == LS.z_on_set([(6, 6), (5, 5)])
    @test test_state.meas_output == test_meas_res
    @test new_state.meas_output == test_meas_res

    new_state = LS.apply(test_notnot, test_state)
    @test test_state.pauli == test_pauli
    @test new_state.pauli == LS.z_on((5, 5)) * LS.x_on((6, 6))
    @test test_state.meas_output == test_meas_res
    @test new_state.meas_output == test_meas_res
end

@testset "Applying Measurements to ComputationalStates" begin
    x_on_meas_qubit = LS.x_on((3, 3))
    z_on_meas_qubit = LS.z_on((3, 3))
    pauli_off_meas_qubit = LS.z_on((5, 5))
    test_meas_res = Dict("a" => 0x00, "b" => 0x01)
    test_meas = LS.Measurement("a", x_on_meas_qubit)

    test_state = LS.ComputationalState(x_on_meas_qubit, test_meas_res)
    new_state = LS.apply(test_meas, test_state)
    @test new_state.meas_output == test_state.meas_output
    @test test_state.pauli == x_on_meas_qubit
    @test new_state.pauli == LS.identity_pauli(LS.nq)

    test_state = LS.ComputationalState(z_on_meas_qubit, test_meas_res)
    new_state = LS.apply(test_meas, test_state)
    @test new_state.meas_output["a"] == 0x01
    @test new_state.meas_output["b"] == 0x01
    @test test_state.pauli == z_on_meas_qubit
    @test new_state.pauli == LS.identity_pauli(LS.nq)

    test_state = LS.ComputationalState(pauli_off_meas_qubit, test_meas_res)
    new_state = LS.apply(test_meas, test_state)
    @test new_state.meas_output["a"] == 0x00
    @test new_state.meas_output["b"] == 0x01
    @test test_state.pauli == pauli_off_meas_qubit
    @test new_state.pauli == pauli_off_meas_qubit
end

@testset "Applying Stabilisers to ComputationalStates" begin
    components = basic_circuit_components()
    test_stabs = components[3]
    test_log_ops = components[4]
    qs = reduce(union, map(LS.support, test_stabs))
    @test issubset(reduce(union, map(LS.support, vcat(test_log_ops...))), qs)
    big_log_ops = [test_log_ops[1][1], test_log_ops[2][1],
                    test_log_ops[1][1] * test_log_ops[2][1]]

    for log_op in big_log_ops
        comp_state = LS.ComputationalState(log_op, Dict{String, UInt8}("a" => 0x00))
        fresh_state = LS.apply_stabs(test_stabs, deepcopy(comp_state))
        @test fresh_state == comp_state
    end

    for q in qs
        for paul in LS.paulis_on(LS.nq, q)
            comp_state = LS.ComputationalState(paul, Dict{String, UInt8}("a" => 0x01, "b" => 0x00))
            fresh_state = LS.apply_stabs(test_stabs, deepcopy(comp_state))
            @test fresh_state == comp_state
        end
    end

    for stab in test_stabs
        comp_state = LS.ComputationalState(stab, Dict{String, UInt8}("a" => 0x01))
        fresh_state = LS.apply_stabs(test_stabs, deepcopy(comp_state))
        @test fresh_state.pauli == LS.identity_pauli(LS.nq)
    end

    comp_state = LS.ComputationalState(test_stabs[1] * LS.z_on((0, 2)), Dict{String, UInt8}())
    fresh_state = LS.apply_stabs(test_stabs, deepcopy(comp_state))
    @test fresh_state.pauli == LS.z_on((0, 2))
end

@testset "circuit_after_fault" begin
    # to cover more code and get multiple layers, we're going to use a
    # different memory gadget
    components = basic_circuit_components()

    gates(tag) = vcat(LS.zzzz_row_meas(tag, (3, 3), 1),
        LS.right_xx_meas(tag, (1, 3), 1),
        LS.left_xx_meas(tag, (5, 3), 1),
        LS.zzzz_row_meas(tag, (3, 3), 2),
        LS.right_xx_meas(tag, (1, 3), 2),
        LS.left_xx_meas(tag, (5, 3), 2))

    stabs = [LS.zzzz_stab((3, 3)),
                LS.right_xx_stab((1, 3)),
                LS.left_xx_stab((5, 3))]

    layers = [LS.Layer(gates("mem_1"), stabs),
                    LS.Layer(gates("mem_2"), stabs)]

    log_ops = ([LS.x_on_set([(2, 2), (4, 2)])],
                    [LS.z_on_set([(2, 2), (2, 4)])])

    default_fault = components[2]

    log_meas_res, ps, log_corr = components[5:7]

    circuit = LS.Circuit(layers, default_fault, stabs, log_ops,
                            log_meas_res, ps, log_corr)

    # circuit's got seven qubits, so there should be 21 input faults
    in_faults = LS.input_faults(circuit)
    @test length(in_faults) == 21
    @test length(LS.qubits(circuit)) == 7

    for f in in_faults
        new_circuit = LS.circuit_after_fault(LS.add_fault(circuit, f))
        @test new_circuit.layers == circuit.layers
        @test new_circuit.fault == f
        @test circuit.fault == default_fault
        @test new_circuit.final_stabs == circuit.final_stabs
        @test new_circuit.log_ops == circuit.log_ops
        @test new_circuit.log_meas_res == circuit.log_meas_res
        @test new_circuit.postselect == circuit.postselect
        @test new_circuit.log_corrections == circuit.log_corrections
    end

    # If we place a fault in the first layer, the second layer should
    # always be included, unchanged, and the length of the first layer
    # should decrease
    id = LS.identity_pauli(LS.nq)
    old_gates = LS.gatelist(LS.layers(circuit)[1])
    n_gates = length(LS.gatelist(LS.layers(circuit)[1]))
    for f_dx = 1:n_gates
        f = LS.Fault(id, 1, f_dx)
        new_circuit = LS.circuit_after_fault(LS.add_fault(circuit, f))
        @test LS.layers(new_circuit)[2] == LS.layers(circuit)[2]
        
        new_gates = LS.gatelist(LS.layers(new_circuit)[1])
        @test new_gates == old_gates[f_dx + 1 : end]
    end
end

@testset "Applying Layers to ComputationalStates" begin
end

@testset "Lattice Surgery Logical Operators" begin
    circuit = LS.lattice_surgery_circuit()
    all_log_ops = vcat(circuit.log_ops...)
    for log_op in all_log_ops
        @test LS.is_logical(circuit, log_op)
    end

    for stab in circuit.final_stabs
        @test ~LS.is_logical(circuit, stab)
    end
end

@testset "Are our circuits d=2 fault-tolerant?" begin
    # @test LS.is_d_2_fault_tolerant(LS.single_memory_circuit())
    # @test LS.is_d_2_fault_tolerant(LS.lattice_surgery_circuit())
end

@testset "Combining ComputationalStates" begin
    fault_1 = LS.Fault(LS.x_on((4, 0)), 1, 0)
    fault_2 = LS.Fault(LS.x_on((6, 0)), 1, 0)
    fault_12 = LS.Fault(LS.x_on_set([(4, 0), (6, 0)]), 1, 0)

    circuit = LS.lattice_surgery_circuit()
    comp_state_1 = LS.run(LS.add_fault(circuit, fault_1))
    comp_state_2 = LS.run(LS.add_fault(circuit, fault_2))
    comp_state_12 = LS.run(LS.add_fault(circuit, fault_12))

    combined_state = LS.combine(comp_state_1, comp_state_2)
    @test combined_state == comp_state_12
end