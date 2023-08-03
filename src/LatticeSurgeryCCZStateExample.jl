#=
	The purpose of this module is to count the malicious fault pairs in
	a lattice surgery implementation of one-bit addition, and estimate
	a logical failure probability based on a uniform error model in
	which each Pauli error after a CNot may occur with probability of
	p/15 and each measurement may return the wrong result with 
	probability p.

	We're going to write out the entire circuit, but replace T gates
	with S gates at the physical level, so the whole thing stays
	Gottesman-Knill-simulable.

	Note: We need to ensure Tomita-Svore ordering. As a test, we ensure
	that the distance-2 surface code has a fault-tolerant syndrome
	measurement gadget.

	It's also important to note that there are some corner measurements
	that, if we interleaved them, would require one ancilla to be used
	to do two measurements at the same time. 
	To get around this, we'll measure differently-coloured tiles in
	different rounds.
	We'll make the generous assumption that there is no memory noise,
	so that this inefficient circuit schedule doesn't affect the
	results.

	There's also the familiar problem of having two weight-two
	stabilizers take up the same ancilla.
	Given that I'm assuming that memory error rates are zero, I can
	stick these measurements into separate layers.
=#
module LatticeSurgeryCCZStateExample

import IterTools as IT
import ProgressMeter as PM
import QuantumClifford as QC

identity_pauli(nq) = QC.PauliOperator(repeat([false], 2 * nq))

include("qubit_layout.jl")
import .QubitLayout: addrs, all_qubits, nq
export addrs, all_qubits, nq

# ------------------- Convenience Functions ------------------------- #

function support(pauli::QC.PauliOperator)
	findall(QC.xbit(pauli) .| QC.zbit(pauli))
end

function weight(pauli::QC.PauliOperator)
	sum(map(count_ones,
		map(tpl -> tpl[1] | tpl[2],
			zip(QC.xview(pauli), QC.zview(pauli)))))
end

function sparse_println(pauli::QC.PauliOperator, qubits=Vector{T}) where T
	function print_ltr(dx)
		x, z = pauli[dx]
		if x && z 
			print("Y_")
		elseif x 
			print("X_")
		elseif z 
			print("Z_")
		end
		print("[", qubits[dx], "] ")
	end

	sprt = support(pauli)
	for dx in sprt 
		print_ltr(dx)
	end
	print("\n")
end

function sparse_println(pauli::QC.PauliOperator)
	sparse_println(pauli, collect(1:pauli.nqubits))
end

# ------------------- Data Structures ------------------------------- #

# const PointT = Tuple{Int, Int}
qubits(gate::QC.sCNOT) = [gate.q1, gate.q2]
qubits(gate::QC.sPhase) = [gate.q]

"""
A fault is a Pauli that occurs before or after a gate in a circuit.
"""
struct Fault
	pauli::QC.PauliOperator
	layer_dx::Int
	gate_dx::Int
end

function Base.:(==)(f1::Fault, f2::Fault)
	(f1.pauli == f2.pauli) && (f1.layer_dx == f2.layer_dx) &&
	(f1.gate_dx == f2.gate_dx)
end

"""
Because we're changing the operators that we measure from layer to
layer, we multiply by the stabilizers of each layer right after we run
that layer.
"""
struct Layer
	gatelist::Vector{Any}
	stabs::Vector{QC.PauliOperator}
end
gatelist(layer::Layer) = layer.gatelist
qubits(layer::Layer) = reduce(union, map(qubits, gatelist(layer)))
qubits(layers::Vector{Layer}) = reduce(union, map(qubits, layers))
stabs(layer::Layer) = layer.stabs
function Base.:(==)(l1::Layer, l2::Layer)
	(l1.gatelist == l2.gatelist) && (l1.stabs == l2.stabs)
end

struct Circuit
	layers::Vector{Layer}
	fault::Fault
	final_stabs::Vector{QC.PauliOperator}
	log_ops::Tuple{Vector{QC.PauliOperator}, Vector{QC.PauliOperator}}
	log_meas_res::Function
	postselect::Function
	log_corrections::Function
end
layers(circuit::Circuit) = circuit.layers
qubits(circuit::Circuit) = reduce(union, map(qubits, layers(circuit)))
fault(circuit::Circuit) = circuit.fault

function add_fault(c, new_fault)
	Circuit(c.layers, new_fault, c.final_stabs, c.log_ops,
			c.log_meas_res, c.postselect, c.log_corrections)
end

struct ComputationalState
	pauli::QC.PauliOperator
	meas_output::Dict{String, UInt8}
end

function ComputationalState(circuit::Circuit)
	meas_output = blank_outputs(circuit)
	pauli = deepcopy(circuit.fault.pauli)

	ComputationalState(pauli, meas_output)
end
function Base.:(==)(cs_1::ComputationalState, cs_2::ComputationalState)
	(cs_1.meas_output == cs_2.meas_output) && (cs_1.pauli == cs_2.pauli)
end

"""
The output from QC.comm is UInt8 and the values are either
0x00 or 0x01.

We start the registers off in 0x00, because any measurement which is
before the fault index doesn't get run, so it would return 0x00.
"""
function blank_outputs(circuit)
	meas_output = Dict{String, UInt8}()
	
	for gate in vcat(map(gatelist, layers(circuit))...)
		if isa(gate, Measurement)
			meas_output[gate.name] = 0x00
		end
	end

	meas_output
end

struct Measurement
	name::String
	pauli::QC.PauliOperator
end
function Base.:(==)(meas_1::Measurement, meas_2::Measurement)
	(meas_1.name == meas_2.name) && (meas_1.pauli == meas_2.pauli)
end

function apply(meas::Measurement, comp_state::ComputationalState)
	pauli, meas_output = deepcopy(comp_state.pauli), comp_state.meas_output
	meas_output[meas.name] = Integer(QC.comm(pauli, meas.pauli))
	# errors on the ancilla don't persist after measurement
	pauli[find_set_bit(meas.pauli)] = (false, false)
	ComputationalState(pauli, meas_output)
end
qubits(meas::Measurement) = [find_set_bit(meas.pauli)]

struct Preparation
	pauli::QC.PauliOperator
end
function Base.:(==)(p1::Preparation, p2::Preparation)
	p1.pauli == p2.pauli
end

function apply(prep::Preparation, comp_state::ComputationalState)
	pauli, meas_output = deepcopy(comp_state.pauli), comp_state.meas_output
	# preparation removes errors
	pauli[find_set_bit(prep.pauli)] = (false, false)
	
	ComputationalState(pauli, meas_output)
end
qubits(prep::Preparation) = [find_set_bit(prep.pauli)]
struct NotNot <: QC.AbstractTwoQubitOperator
	q1::Int
	q2::Int
end
qubits(gate::NotNot) = [gate.q1, gate.q2]

function QC.qubit_kernel(gate::NotNot, x1, z1, x2, z2)
	new_phase = ~iszero( (z1 & z2) & (xor(x1, x2)) )

	(xor(z2, x1), z1, xor(x2, z1), z2, new_phase)
end

# ------------------------ Utility Functions ------------------------ #

"""
`find_set_bit(weight_one_pauli::QC.PauliOperator)`

I can't find out how to get the qubits that a QC.Pauli affects, so I'm
hacking something together.
"""
function find_set_bit(weight_one_pauli::QC.PauliOperator)
	n_zeros = map(trailing_zeros, weight_one_pauli.xz)
	n_ints = length(n_zeros)
	half = n_ints >> 1
	
	set_bit_dx = 1
	if all(n_zeros[1:half] .== 64)
		# the Pauli's a Z
		for elem in n_zeros[half + 1 : end]
			set_bit_dx += elem
			if elem < 64 
				break
			end
		end
	else
		for elem in n_zeros[1 : half]
			set_bit_dx += elem
			if elem < 64 
				break
			end
		end
	end

	set_bit_dx
end

"""
`circuit_after_fault(circuit::Circuit)`

We include every gate after the fault in the specified layer, and every
layer thereafter.
"""
function circuit_after_fault(c::Circuit)
	f = fault(c)
	layrrs = layers(c)
	layrr = layrrs[f.layer_dx]
	sublayer = Layer(gatelist(layrr)[f.gate_dx + 1 : end], stabs(layrr))

	Circuit(vcat(sublayer, layrrs[f.layer_dx + 1 : end]), f,
				c.final_stabs, c.log_ops, c.log_meas_res, c.postselect, c.log_corrections)
end

"""
`apply(stabs::Vector{QC.PauliOperator}, comp_state)`

Errors that occur during stabilizer measurement are equivalent to
products of stabilizers with those errors (note, these may look like
logicals when compared with stabilizers measured later on).

Fortunately, error propagation is limited such that there exists at
most one stabilizer (and it's a generator). 
"""
function apply_stabs(stabs::Vector{T}, comp_state::ComputationalState) where T <: QC.PauliOperator
	current_pauli = comp_state.pauli
	
	for stab in stabs
		new_pauli = stab * current_pauli
		if weight(new_pauli) < weight(current_pauli)
			current_pauli = new_pauli
		end
	end

	ComputationalState(current_pauli, comp_state.meas_output)
end

function apply(layer::Layer, comp_state::ComputationalState)
	for gate in gatelist(layer)
		comp_state = apply(gate, comp_state)
	end

	apply_stabs(stabs(layer), comp_state)
end

function apply_gate_to_pauli(gate, pauli::QC.PauliOperator)
	QC.apply!(QC.Stabilizer([pauli]), gate)[1]
end

function apply(gate, comp_state::ComputationalState)
	new_pauli = apply_gate_to_pauli(gate, comp_state.pauli)
	ComputationalState(new_pauli, comp_state.meas_output)
end

"""
`run(circuit::Circuit)`

Returns an array of measurement results and a Pauli output on the final
data qubits that can be used to tell whether a logical error has
occurred without being detected.
"""
function run(circuit::Circuit)
	comp_state = ComputationalState(circuit)
	
	for layer in layers(circuit_after_fault(circuit))
		comp_state = apply(layer, comp_state)
	end

	comp_state
end

"""
`cnot(ctrl, targ)`

uses the `addrs` from `qubit_layout.jl` to produce an integer-indexed
CNot.
"""
cnot(ctrl, targ) = QC.sCNOT(addrs[ctrl], addrs[targ])

"""
`s(q)`

uses the `addrs` from `qubit_layout.jl` to produce an integer-indexed
S gate.
"""
s(q) = QC.sPhase(addrs[q])

"""
`notnot(ctrl, targ)`

uses the `addrs` from `qubit_layout.jl` to produce an integer-indexed
"notnot" gate, which acts as (H otimes H) CZ (H otimes H). 
"""
notnot(ctrl, targ) = NotNot(addrs[ctrl], addrs[targ])

z_on(anc) = QC.single_z(nq, addrs[anc])
x_on(anc) = QC.single_x(nq, addrs[anc])

x_on_set(st) = prod(map(x_on, st))
z_on_set(st) = prod(map(z_on, st))
xz_on_set(xs, zs) = x_on_set(xs) * z_on_set(zs)

# --------------------- Circuit Construction ------------------------ #

"""
`d_2_repetition_code_circuit()`

In order to test the fault-counting function, we need a circuit with
few enough gates that we can count the faults 'by hand' (though I've
done as much as I can to classify the faults into equivalent sets in
order to limit the amount of arithmetic). We do this in an
accompanying file, `manual_fault_count.txt`.  
"""
function d_2_repetition_code_circuit()
	tag = "rep_code"
	anc = (5, 5)
	gates = vcat(down_zz_meas(tag, anc, 1), down_zz_meas(tag, anc, 2))
	stabs = [down_zz_stab(anc)]
	default_fault = Fault(identity_pauli(nq), 1, 0)
	log_ops = ([z_on((4, 4))], [x_on_set([(4, 4), (6, 4)])])

	function log_meas_res(comp_state::ComputationalState)
		Dict{String, UInt8}()
	end

	function postselect(comp_state::ComputationalState, log_results)
		false
	end

	function log_corrections(comp_state, log_results)
		comp_state
	end

	Circuit([Layer(gates, stabs)], default_fault, stabs, log_ops,
				log_meas_res, postselect, log_corrections)
end

"""
`postselected_d_2_repetition_code_circuit()`

In this version of the repetition code circuit, we postselect on any
non-zero syndrome. 
"""
function postselected_d_2_repetition_code_circuit()
	tag = "rep_code"
	anc = (5, 5)
	gates = vcat(down_zz_meas(tag, anc, 1), down_zz_meas(tag, anc, 2))
	stabs = [down_zz_stab(anc)]
	default_fault = Fault(identity_pauli(nq), 1, 0)
	log_ops = ([z_on((4, 4))], [x_on_set([(4, 4), (6, 4)])])

	function log_meas_res(comp_state::ComputationalState)
		Dict{String, UInt8}()
	end

	function postselect(comp_state::ComputationalState, log_results)
		any(val == 0x01 for val in values(comp_state.meas_output))
	end

	function log_corrections(comp_state, log_results)
		comp_state
	end

	Circuit([Layer(gates, stabs)], default_fault, stabs, log_ops,
				log_meas_res, postselect, log_corrections)
end

"""
`single_memory_circuit()`

For testing and profiling purposes, we do a single memory gadget for
the distance-2 surface code.
"""
function single_memory_circuit()
	tag = "mem"
	
	gates = vcat(zzzz_meas(tag, (1, 3), 1),
		down_xx_meas(tag, (1, 5), 1),
		up_xx_meas(tag, (1, 1), 1),
		zzzz_meas(tag, (1, 3), 2),
		down_xx_meas(tag, (1, 5), 2),
		up_xx_meas(tag, (1, 1), 2))

	stabs = [zzzz_stab((1, 3)),
				down_xx_stab((1, 5)),
				up_xx_stab((1, 1))]

	default_fault = Fault(identity_pauli(nq), 1, 0)

	log_ops = ([z_on_set([(0, 4), (2, 4)])], [x_on_set([(0, 2), (0, 4)])])
	
	function log_meas_res(comp_state::ComputationalState)
		Dict{String, UInt8}()
	end

	function postselect(comp_state::ComputationalState, log_results)
		any(val == 0x01 for val in values(comp_state.meas_output))
	end

	function log_corrections(comp_state, log_results)
		comp_state
	end

	Circuit([Layer(gates, stabs)], default_fault, stabs, log_ops,
				log_meas_res, postselect, log_corrections)
end

function lattice_surgery_circuit()
	
	layers = [
				Layer(x_1_x_abcd_meas(), x_1_x_abcd_stabs()),
				Layer(x_abcdefgh_meas(), x_abcdefgh_stabs()),
				Layer(x_3_x_aceg_meas(), x_3_x_aceg_stabs()),
				Layer(x_2_x_abef_meas(), x_2_x_abef_stabs()),
				Layer(s_x_meas(), Vector{QC.PauliOperator}())
			]
	
	default_fault = Fault(identity_pauli(nq), 1, 0)
	
	final_stabs = output_stabilizers()
	
	log_ops = logical_operators()

	log_meas_res = logical_measurement_results

	postselect = is_postselected

	log_corrections = do_logical_corrections

	Circuit(layers, default_fault, final_stabs, log_ops,
			log_meas_res, postselect, log_corrections)
end

function measurement_name(tag, stab, anc, round)
	join(map(string, [tag, stab, anc, round]), "_")
end

function zzzz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zzzz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (1, 1), anc), cnot(anc .+ (1, -1), anc),
		cnot(anc .+ (-1, 1), anc), cnot(anc .+ (-1, -1), anc), 
		Measurement(meas_name, z_on(anc))]
end

function zzzz_stab(anc)
	z_on_set([anc .+ (1, 1), anc .+ (1, -1),
				anc .+ (-1, 1), anc .+ (-1, -1)])
end

function zzzz_row_meas(tag, anc, round)
	# don't need a different name, because ZZZZ measurement is never
	# carried out twice on the same ancilla in the same circuit.
	meas_name = measurement_name(tag, "zzzz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (1, 1), anc), cnot(anc .+ (-1, 1), anc),
		cnot(anc .+ (1, -1), anc), cnot(anc .+ (-1, -1), anc), 
		Measurement(meas_name, z_on(anc))]
end

function xxxx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "xxxx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (1, 1)), cnot(anc, anc .+ (1, -1)),
		cnot(anc, anc .+ (-1, 1)), cnot(anc, anc .+ (-1, -1)), 
		Measurement(meas_name, x_on(anc))]
end

function xxxx_stab(anc)
	x_on_set([anc .+ (1, 1), anc .+ (1, -1),
				anc .+ (-1, 1), anc .+ (-1, -1)])
end

function se_xxx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "xxx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (1, 1)),
		cnot(anc, anc .+ (-1, 1)), cnot(anc, anc .+ (-1, -1)), 
		Measurement(meas_name, x_on(anc))]
end

function se_xxx_stab(anc)
	x_on_set([anc .+ (1, 1), anc .+ (-1, 1), anc .+ (-1, -1)])
end

function sw_xxx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "xxx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (1, 1)),
		cnot(anc, anc .+ (-1, 1)), cnot(anc, anc .+ (1, -1)), 
		Measurement(meas_name, x_on(anc))]
end

function sw_xxx_stab(anc)
	x_on_set([anc .+ (1, 1), anc .+ (-1, 1), anc .+ (1, -1)])
end

function ne_xxx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "xxx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (1, -1)),
		cnot(anc, anc .+ (-1, 1)), cnot(anc, anc .+ (-1, -1)), 
		Measurement(meas_name, x_on(anc))]
end

function ne_xxx_stab(anc)
	x_on_set([anc .+ (1, -1), anc .+ (-1, 1), anc .+ (-1, -1)])
end

function up_xx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "up_xx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (1, 1)), cnot(anc, anc .+ (-1, 1)),
		Measurement(meas_name, x_on(anc))]
end

function up_xx_stab(anc)
	x_on_set([anc .+ (1, 1), anc .+ (-1, 1)])
end

function zxzx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zxzx", anc, round)
	# note, shifts have to go top row bottom row here
	[Preparation(z_on(anc)), cnot(anc .+ (1, 1), anc), notnot(anc .+ (-1, 1), anc),
		cnot(anc .+ (1, -1), anc), notnot(anc .+ (-1, -1), anc), 
		Measurement(meas_name, z_on(anc))]
end

function zxzx_stab(anc)
	xz_on_set([anc .+ (1, 1), anc .+ (1, -1)],
				[anc .+ (-1, 1), anc .+ (-1, -1)])
end

function zxzx_col_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zxzx_col", anc, round)
	# note, shifts have to go right column left column here
	[Preparation(z_on(anc)), cnot(anc .+ (1, 1), anc), notnot(anc .+ (1, -1), anc),
		cnot(anc .+ (-1, 1), anc), notnot(anc .+ (-1, -1), anc), 
		Measurement(meas_name, z_on(anc))]
end

function zxzx_col_stab(anc)
	xz_on_set([anc .+ (1, -1), anc .+ (-1, -1)],
		[anc .+ (1, 1), anc .+ (-1, 1)])
end

function xzxz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "xzxz", anc, round)
	# note, shifts have to go top row bottom row here
	[Preparation(z_on(anc)), notnot(anc .+ (1, 1), anc), cnot(anc .+ (-1, 1), anc),
		notnot(anc .+ (1, -1), anc), cnot(anc .+ (-1, -1), anc), 
		Measurement(meas_name, z_on(anc))]
end

function xzxz_stab(anc)
	xz_on_set([anc .+ (1, 1), anc .+ (1, -1)],
				[anc .+ (-1, 1), anc .+ (-1, -1)])
end

function left_zz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (-1, 1), anc), cnot(anc .+ (-1, -1), anc),
		Measurement(meas_name, z_on(anc))]
end

function left_zz_stab(anc)
	z_on_set([anc .+ (-1, 1), anc .+ (-1, -1)])
end

function right_zz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (1, 1), anc), cnot(anc .+ (1, -1), anc),
		Measurement(meas_name, z_on(anc))]
end

function right_zz_stab(anc)
	z_on_set([anc .+ (1, 1), anc .+ (1, -1)])
end

function down_xx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "down_xx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (1, -1)), cnot(anc, anc .+ (-1, -1)),
		Measurement(meas_name, x_on(anc))]
end

function down_xx_stab(anc)
	x_on_set([anc .+ (-1, -1), anc .+ (1, -1)])
end

function right_xx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "xx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (1, 1)), cnot(anc, anc .+ (1, -1)),
		Measurement(meas_name, x_on(anc))]
end

function right_xx_stab(anc)
	x_on_set([anc .+ (1, 1), anc .+ (1, -1)])
end

function left_xx_meas(tag, anc, round)
	meas_name = measurement_name(tag, "xx", anc, round)
	[Preparation(x_on(anc)), cnot(anc, anc .+ (-1, 1)), cnot(anc, anc .+ (-1, -1)),
		Measurement(meas_name, x_on(anc))]
end

function left_xx_stab(anc)
	x_on_set([anc .+ (-1, 1), anc .+ (-1, -1)])
end

function down_zz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (1, -1), anc), cnot(anc .+ (-1, -1), anc),
		Measurement(meas_name, z_on(anc))]
end

function down_zz_stab(anc)
	z_on_set([anc .+ (1, -1), anc .+ (-1, -1)])
end

function up_zz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (1, 1), anc), cnot(anc .+ (-1, 1), anc),
		Measurement(meas_name, z_on(anc))]
end

function up_zz_stab(anc)
	z_on_set([anc .+ (1, 1), anc .+ (-1, 1)])
end

function nw_zz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (1, 1), anc), cnot(anc .+ (-1, -1), anc),
		Measurement(meas_name, z_on(anc))]
end

function nw_zz_stab(anc)
	z_on_set([anc .+ (1, 1), anc .+ (-1, -1)])
end

function sw_zz_meas(tag, anc, round)
	meas_name = measurement_name(tag, "zz", anc, round)
	[Preparation(z_on(anc)), cnot(anc .+ (1, -1), anc), cnot(anc .+ (-1, 1), anc),
		Measurement(meas_name, z_on(anc))]
end

function sw_zz_stab(anc)
	z_on_set([anc .+ (1, -1), anc .+ (-1, 1)])
end

function measurement_subcircuits(tag)
	xxxx_1(anc) = xxxx_meas(tag, anc, 1)
	xxxx_2(anc) = xxxx_meas(tag, anc, 2)

	se_xxx_1(anc) = se_xxx_meas(tag, anc, 1)
	se_xxx_2(anc) = se_xxx_meas(tag, anc, 2)

	sw_xxx_1(anc) = sw_xxx_meas(tag, anc, 1)
	sw_xxx_2(anc) = sw_xxx_meas(tag, anc, 2)

	ne_xxx_1(anc) = ne_xxx_meas(tag, anc, 1)
	ne_xxx_2(anc) = ne_xxx_meas(tag, anc, 2)

	zzzz_1(anc) = zzzz_meas(tag, anc, 1)
	zzzz_2(anc) = zzzz_meas(tag, anc, 2)

	zzzz_row_1(anc) = zzzz_row_meas(tag, anc, 1)
	zzzz_row_2(anc) = zzzz_row_meas(tag, anc, 2)

	xzxz_1(anc) = xzxz_meas(tag, anc, 1)
	xzxz_2(anc) = xzxz_meas(tag, anc, 2)

	zxzx_1(anc) = zxzx_meas(tag, anc, 1)
	zxzx_2(anc) = zxzx_meas(tag, anc, 2)

	zxzx_col_1(anc) = zxzx_col_meas(tag, anc, 1)
	zxzx_col_2(anc) = zxzx_col_meas(tag, anc, 2)
	
	up_xx_1(anc) = up_xx_meas(tag, anc, 1)
	up_xx_2(anc) = up_xx_meas(tag, anc, 2)

	up_zz_1(anc) = up_zz_meas(tag, anc, 1)
	up_zz_2(anc) = up_zz_meas(tag, anc, 2)

	nw_zz_1(anc) = nw_zz_meas(tag, anc, 1)
	nw_zz_2(anc) = nw_zz_meas(tag, anc, 2)

	sw_zz_1(anc) = sw_zz_meas(tag, anc, 1)
	sw_zz_2(anc) = sw_zz_meas(tag, anc, 2)

	down_xx_1(anc) = down_xx_meas(tag, anc, 1)
	down_xx_2(anc) = down_xx_meas(tag, anc, 2)

	down_zz_1(anc) = down_zz_meas(tag, anc, 1)
	down_zz_2(anc) = down_zz_meas(tag, anc, 2)

	left_xx_1(anc) = left_xx_meas(tag, anc, 1)
	left_xx_2(anc) = left_xx_meas(tag, anc, 2)

	left_zz_1(anc) = left_zz_meas(tag, anc, 1)
	left_zz_2(anc) = left_zz_meas(tag, anc, 2)

	right_xx_1(anc) = right_xx_meas(tag, anc, 1)
	right_xx_2(anc) = right_xx_meas(tag, anc, 2)
	
	right_zz_1(anc) = right_zz_meas(tag, anc, 1)
	right_zz_2(anc) = right_zz_meas(tag, anc, 2)
	
	subcircuits = Dict(
		"xxxx_1" => xxxx_1, "xxxx_2" => xxxx_2, "zzzz_1" => zzzz_1,
		"se_xxx_1" => se_xxx_1, "se_xxx_2" => se_xxx_2,
		"sw_xxx_1" => sw_xxx_1, "sw_xxx_2" => sw_xxx_2,
		"ne_xxx_1" => ne_xxx_1, "ne_xxx_2" => ne_xxx_2,
		"zzzz_2" => zzzz_2, "zzzz_row_1" => zzzz_row_1,
		"zzzz_row_2" => zzzz_row_2, "xzxz_1" => xzxz_1,
		"xzxz_2" => xzxz_2, "zxzx_1" => zxzx_1, "zxzx_2" => zxzx_2,
		"zxzx_col_1" => zxzx_col_1, "zxzx_col_2" => zxzx_col_2,
		"up_xx_1" => up_xx_1, "up_xx_2" => up_xx_2,
		"up_zz_1" => up_zz_1, "up_zz_2" => up_zz_2,
		"nw_zz_1" => nw_zz_1, "nw_zz_2" => nw_zz_2,
		"sw_zz_1" => sw_zz_1, "sw_zz_2" => sw_zz_2,
		"down_xx_1" => down_xx_1, "down_xx_2" => down_xx_2,
		"down_zz_1" => down_zz_1, "down_zz_2" => down_zz_2,
		"left_xx_1" => left_xx_1, "left_xx_2" => left_xx_2,
		"left_zz_1" => left_zz_1, "left_zz_2" => left_zz_2,
		"right_xx_1" => right_xx_1, "right_xx_2" => right_xx_2,
		"right_zz_1" => right_zz_1, "right_zz_2" => right_zz_2
		)

	subcircuits
end

function x_1_x_abcd_ancs()
	layer_1_zzzz = [(1, 5), (1, 9), (1, 13), (1, 17), (1, 21)]
	layer_1_xxxx = [(5, 9), (5, 13), (5, 17)]
	layer_1_se_xxx = [(5, 5)]
	layer_1_ne_xxx = [(5, 21)]

	layer_2_up_xx = [(1, 3), (1, 7), (1, 11), (1, 15), (1, 19)]
	layer_2_zxzx = [(3, 5), (3, 9), (3, 13), (3, 17), (3, 21)]
	layer_2_zzzz = [(5, 7), (5, 11), (5, 15), (5, 19)]
	layer_2_left_zz = [(7, 9), (7, 13), (7, 17)]
	layer_2_nw_zz = [(5, 5)]
	layer_2_sw_zz = [(5, 21)]

	layer_3_down_xx = [(1, 7), (1, 11), (1, 15), (1, 19), (1, 23)]

	Dict("layer_1_zzzz" => layer_1_zzzz,
			"layer_1_xxxx" => layer_1_xxxx,
			"layer_1_se_xxx" => layer_1_se_xxx,
			"layer_1_ne_xxx" => layer_1_ne_xxx,
			"layer_2_up_xx" => layer_2_up_xx,
			"layer_2_zxzx" => layer_2_zxzx,
			"layer_2_zzzz" => layer_2_zzzz,
			"layer_2_left_zz" => layer_2_left_zz,
			"layer_2_nw_zz" => layer_2_nw_zz,
			"layer_2_sw_zz" => layer_2_sw_zz,
			"layer_3_down_xx" => layer_3_down_xx)
end

function x_1_x_abcd_meas()
	ancs = x_1_x_abcd_ancs()

	tag = "x_1_x_abcd"
	circs = measurement_subcircuits(tag)
	
	layer_1 = vcat(
		vcat(map(circs["zzzz_1"], ancs["layer_1_zzzz"])...), 
		vcat(map(circs["xxxx_1"], ancs["layer_1_xxxx"])...),
		vcat(map(circs["se_xxx_1"], ancs["layer_1_se_xxx"])...),
		vcat(map(circs["ne_xxx_1"], ancs["layer_1_ne_xxx"])...)
		)

	layer_2 = vcat(
		vcat(map(circs["up_xx_1"], ancs["layer_2_up_xx"])...),
		vcat(map(circs["zxzx_1"], ancs["layer_2_zxzx"])...), 
		vcat(map(circs["zzzz_1"], ancs["layer_2_zzzz"])...),
		vcat(map(circs["left_zz_1"], ancs["layer_2_left_zz"])...),
		vcat(map(circs["sw_zz_1"], ancs["layer_2_sw_zz"])...),
		vcat(map(circs["nw_zz_1"], ancs["layer_2_nw_zz"])...)
		)

	layer_3 = vcat(map(circs["down_xx_1"], ancs["layer_3_down_xx"])...)

	layer_4 = vcat(
		vcat(map(circs["zzzz_2"], ancs["layer_1_zzzz"])...), 
		vcat(map(circs["xxxx_2"], ancs["layer_1_xxxx"])...),
		vcat(map(circs["se_xxx_2"], ancs["layer_1_se_xxx"])...),
		vcat(map(circs["ne_xxx_2"], ancs["layer_1_ne_xxx"])...)
		)

	layer_5 = vcat(
		vcat(map(circs["up_xx_2"], ancs["layer_2_up_xx"])...),
		vcat(map(circs["zxzx_2"], ancs["layer_2_zxzx"])...), 
		vcat(map(circs["zzzz_2"], ancs["layer_2_zzzz"])...),
		vcat(map(circs["left_zz_2"], ancs["layer_2_left_zz"])...),
		vcat(map(circs["sw_zz_2"], ancs["layer_2_sw_zz"])...),
		vcat(map(circs["nw_zz_2"], ancs["layer_2_nw_zz"])...)
		)

	layer_6 = vcat(map(circs["down_xx_2"], ancs["layer_3_down_xx"])...)

	vcat(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6)
end

function x_1_x_abcd_stabs()
	ancs = x_1_x_abcd_ancs()
	
	zzzz = map(zzzz_stab, ancs["layer_1_zzzz"])
	xxxx = map(xxxx_stab, ancs["layer_1_xxxx"])
	se_xxx = map(se_xxx_stab, ancs["layer_1_se_xxx"])
	ne_xxx = map(ne_xxx_stab, ancs["layer_1_ne_xxx"])
	up_xx = map(up_xx_stab, ancs["layer_2_up_xx"])
	zxzx = map(zxzx_stab, ancs["layer_2_zxzx"])
	zzzz = vcat(zzzz, map(zzzz_stab, ancs["layer_2_zzzz"]))
	left_zz = map(left_zz_stab, ancs["layer_2_left_zz"])
	nw_zz = map(nw_zz_stab, ancs["layer_2_nw_zz"])
	sw_zz = map(sw_zz_stab, ancs["layer_2_sw_zz"])
	down_xx = map(down_xx_stab, ancs["layer_3_down_xx"])

	vcat(zzzz, xxxx, se_xxx, ne_xxx, up_xx,
			zxzx, zzzz, left_zz, nw_zz, sw_zz, down_xx)
end

function x_abcdefgh_ancs()
	layer_1_zzzz = [(1, 9), (1, 13), (1, 17), (1, 21),
					(9, 9), (9, 13), (9, 17), (9, 21)]
	
	layer_1_xxxx = [(5, 9), (5, 13), (5, 17), (5, 21)]

	layer_2_zxzx = [(3, 9), (3, 13), (3, 17), (3, 21)]
	layer_2_xzxz = [(7, 9), (7, 13), (7, 17), (7, 21)]

	layer_3_down_xx = [(1, 11), (1, 15), (1, 19), (1, 23),
						(9, 11), (9, 15), (9, 19), (9, 23)]
	layer_3_down_zz = [(5, 23)]
	layer_3_zzzz = [(5, 11), (5, 15), (5, 19)]

	layer_4_up_xx = [(1, 7), (1, 11), (1, 15), (1, 19),
						(9, 7), (9, 11), (9, 15), (9, 19)]
	layer_4_up_zz = [(5, 7)]

	Dict(
			"layer_1_zzzz" => layer_1_zzzz,
			"layer_1_xxxx" => layer_1_xxxx,
			"layer_2_zxzx" => layer_2_zxzx,
			"layer_2_xzxz" => layer_2_xzxz,
			"layer_3_down_xx" => layer_3_down_xx,
			"layer_3_down_zz" => layer_3_down_zz,
			"layer_3_zzzz" => layer_3_zzzz,
			"layer_4_up_xx" => layer_4_up_xx,
			"layer_4_up_zz" => layer_4_up_zz
		)
end

function x_abcdefgh_meas()
	ancs = x_abcdefgh_ancs()
	tag = "x_abcdefgh"
	circs = measurement_subcircuits(tag)

	layer_1 = vcat(
		vcat(map(circs["zzzz_1"], ancs["layer_1_zzzz"])...),
		vcat(map(circs["xxxx_1"], ancs["layer_1_xxxx"])...)
		)

	layer_2 = vcat(
		vcat(map(circs["zxzx_1"], ancs["layer_2_zxzx"])...),
		vcat(map(circs["xzxz_1"], ancs["layer_2_xzxz"])...)
		)

	layer_3 = vcat(
		vcat(map(circs["down_xx_1"], ancs["layer_3_down_xx"])...),
		vcat(map(circs["down_zz_1"], ancs["layer_3_down_zz"])...),
		vcat(map(circs["zzzz_1"], ancs["layer_3_zzzz"])...)
		)

	layer_4 = vcat(
		vcat(map(circs["up_xx_1"], ancs["layer_4_up_xx"])...),
		vcat(map(circs["up_zz_1"], ancs["layer_4_up_zz"])...)
		)
	
	layer_5 = vcat(
		vcat(map(circs["zzzz_2"], ancs["layer_1_zzzz"])...),
		vcat(map(circs["xxxx_2"], ancs["layer_1_xxxx"])...)
		)

	layer_6 = vcat(
		vcat(map(circs["zxzx_2"], ancs["layer_2_zxzx"])...),
		vcat(map(circs["xzxz_2"], ancs["layer_2_xzxz"])...)
		)

	layer_7 = vcat(
		vcat(map(circs["down_xx_2"], ancs["layer_3_down_xx"])...),
		vcat(map(circs["down_zz_2"], ancs["layer_3_down_zz"])...),
		vcat(map(circs["zzzz_2"], ancs["layer_3_zzzz"])...)
		)

	layer_8 = vcat(
		vcat(map(circs["up_xx_2"], ancs["layer_4_up_xx"])...),
		vcat(map(circs["up_zz_2"], ancs["layer_4_up_zz"])...)
		)

	vcat(layer_1, layer_2, layer_3, layer_4,
			layer_5, layer_6, layer_7, layer_8)
end

function x_abcdefgh_stabs()
	ancs = x_abcdefgh_ancs()
	
	zzzz = map(zzzz_stab, ancs["layer_1_zzzz"])
	xxxx = map(xxxx_stab, ancs["layer_1_xxxx"])
	zxzx = map(zxzx_stab, ancs["layer_2_zxzx"])
	xzxz = map(xzxz_stab, ancs["layer_2_xzxz"])
	down_xx = map(down_xx_stab, ancs["layer_3_down_xx"])
	down_zz = map(down_zz_stab, ancs["layer_3_down_zz"])
	zzzz = vcat(zzzz, map(zzzz_stab, ancs["layer_3_zzzz"]))

	up_xx = map(up_xx_stab, ancs["layer_4_up_xx"])
	up_zz = map(up_zz_stab, ancs["layer_4_up_zz"])
	
	vcat(zzzz, xxxx, zxzx, xzxz, down_xx, down_zz, up_xx, up_zz)
end

function x_3_x_aceg_ancs()
	layer_1_zzzz = [(1, 9), (1, 13), (1, 17), (1, 21),
						(9, 5), (9, 9), (9, 13), (9, 17), (9, 21)]
	layer_1_xxxx = [(5, 9), (5, 13), (5, 17), (5, 21)]
	layer_1_sw_xxx = [(5, 5)]

	layer_2_xzxz = [(7, 5), (7, 13), (7, 21)]
	layer_2_zxzx = [(3, 13), (3, 21)]

	layer_3_down_xx = [(1, 11), (1, 15), (1, 19), (1, 23),
							(9, 7), (9, 11), (9, 15), (9, 19), (9, 23)]
	layer_3_down_zz = [(5, 23)]
	layer_3_zzzz = [(5, 7), (5, 11), (5, 15), (5, 19)]

	layer_4_up_xx = [(1, 7), (1, 11), (1, 15), (1, 19),
							(9, 3), (9, 7), (9, 11), (9, 15), (9, 19)]
	layer_4_right_zz = [(3, 9), (3, 17)]
	layer_4_left_zz = [(7, 9), (7, 17)]
	layer_4_sw_zz = [(5, 5)]

	Dict(
			"layer_1_zzzz" => layer_1_zzzz,
			"layer_1_xxxx" => layer_1_xxxx,
			"layer_1_sw_xxx" => layer_1_sw_xxx,
			"layer_2_xzxz" => layer_2_xzxz,
			"layer_2_zxzx" => layer_2_zxzx,
			"layer_3_down_xx" => layer_3_down_xx,
			"layer_3_down_zz" => layer_3_down_zz,
			"layer_3_zzzz" => layer_3_zzzz,
			"layer_4_up_xx" => layer_4_up_xx,
			"layer_4_right_zz" => layer_4_right_zz,
			"layer_4_left_zz" => layer_4_left_zz,
			"layer_4_sw_zz" => layer_4_sw_zz
		)
end

function x_3_x_aceg_meas()
	ancs = x_3_x_aceg_ancs()
	tag = "x_3_x_aceg"
	circs = measurement_subcircuits(tag)

	layer_1 = vcat(
		vcat(map(circs["zzzz_1"], ancs["layer_1_zzzz"])...),
		vcat(map(circs["xxxx_1"], ancs["layer_1_xxxx"])...),
		vcat(map(circs["sw_xxx_1"], ancs["layer_1_sw_xxx"])...)
		)

	layer_2 = vcat(
		vcat(map(circs["xzxz_1"], ancs["layer_2_xzxz"])...),
		vcat(map(circs["zxzx_1"], ancs["layer_2_zxzx"])...)
		)

	layer_3 = vcat(
		vcat(map(circs["down_xx_1"], ancs["layer_3_down_xx"])...),
		vcat(map(circs["down_zz_1"], ancs["layer_3_down_zz"])...),
		vcat(map(circs["zzzz_1"], ancs["layer_3_zzzz"])...)
		)

	layer_4 = vcat(
		vcat(map(circs["up_xx_1"], ancs["layer_4_up_xx"])...),
		vcat(map(circs["sw_zz_1"], ancs["layer_4_sw_zz"])...),
		vcat(map(circs["left_zz_1"], ancs["layer_4_left_zz"])...),
		vcat(map(circs["right_zz_1"], ancs["layer_4_right_zz"])...))

	layer_5 = vcat(
		vcat(map(circs["zzzz_2"], ancs["layer_1_zzzz"])...),
		vcat(map(circs["xxxx_2"], ancs["layer_1_xxxx"])...),
		vcat(map(circs["sw_xxx_2"], ancs["layer_1_sw_xxx"])...)
		)

	layer_6 = vcat(
		vcat(map(circs["xzxz_2"], ancs["layer_2_xzxz"])...),
		vcat(map(circs["zxzx_2"], ancs["layer_2_zxzx"])...)
		)

	layer_7 = vcat(
		vcat(map(circs["down_xx_2"], ancs["layer_3_down_xx"])...),
		vcat(map(circs["down_zz_2"], ancs["layer_3_down_zz"])...),
		vcat(map(circs["zzzz_2"], ancs["layer_3_zzzz"])...)
		)

	layer_8 = vcat(
		vcat(map(circs["up_xx_2"], ancs["layer_4_up_xx"])...),
		vcat(map(circs["sw_zz_2"], ancs["layer_4_sw_zz"])...),
		vcat(map(circs["left_zz_2"], ancs["layer_4_left_zz"])...),
		vcat(map(circs["right_zz_2"], ancs["layer_4_right_zz"])...))

	vcat(layer_1, layer_2, layer_3, layer_4,
			layer_5, layer_6, layer_7, layer_8)
end

function x_3_x_aceg_stabs()
	ancs = x_3_x_aceg_ancs()
	
	zzzz = map(zzzz_stab, ancs["layer_1_zzzz"])
	xxxx = map(xxxx_stab, ancs["layer_1_xxxx"])
	sw_xxx = map(xxxx_stab, ancs["layer_1_sw_xxx"])

	xzxz = map(xzxz_stab, ancs["layer_2_xzxz"])
	zxzx = map(zxzx_stab, ancs["layer_2_zxzx"])
	
	down_xx = map(down_xx_stab, ancs["layer_3_down_xx"])
	down_zz = map(down_zz_stab, ancs["layer_3_down_zz"])
	zzzz = vcat(zzzz, map(zzzz_stab, ancs["layer_3_zzzz"]))
	
	right_zz = map(right_zz_stab, ancs["layer_4_right_zz"])
	left_zz = map(left_zz_stab, ancs["layer_4_left_zz"])
	sw_zz = map(sw_zz_stab, ancs["layer_4_sw_zz"])

	vcat(zzzz, xxxx, xzxz, zxzx, down_xx,
			down_zz, right_zz, left_zz, sw_zz)
end

function x_2_x_abef_ancs()
	layer_1_zzzz = [(1, 9), (1, 13), (1, 17), (1, 21),
					(9, 9), (9, 13), (9, 17), (9, 21)]
	layer_1_xxxx = [(5, 5), (5, 9), (5, 13), (5, 17), (5, 21)]
	layer_1_zzzz_row = [(5, 1)]

	layer_2_down_zz = [(5, 23)]
	layer_2_down_xx = [(1, 11), (1, 15), (1, 19), (1, 23),
						(9, 11), (9, 15), (9, 19), (9, 23)]
	layer_2_zxzx_col = [(5, 3)]
	layer_2_zzzz = [(5, 7), (5, 11), (5, 15), (5, 19)]

	layer_3_up_xx = [(1, 7), (1, 11), (1, 15), (1, 19),
							(9, 7), (9, 11), (9, 15), (9, 19)]
	layer_3_left_xx = [(7, 1)]
	layer_3_right_xx = [(3, 1)]
	layer_3_left_zz = [(7, 5), (7, 9), (7, 13)]
	layer_3_right_zz = [(3, 5), (3, 9), (3, 13)]

	layer_4_xzxz = [(7, 17), (7, 21)]
	layer_4_zxzx = [(3, 17), (3, 21)]

	Dict(
		"layer_1_zzzz" => layer_1_zzzz,
		"layer_1_xxxx" => layer_1_xxxx,
		"layer_1_zzzz_row" => layer_1_zzzz_row,
		"layer_2_down_zz" => layer_2_down_zz,
		"layer_2_down_xx" => layer_2_down_xx,
		"layer_2_zxzx_col" => layer_2_zxzx_col,
		"layer_2_zzzz" => layer_2_zzzz,
		"layer_3_up_xx" => layer_3_up_xx,
		"layer_3_left_xx" => layer_3_left_xx,
		"layer_3_right_xx" => layer_3_right_xx,
		"layer_3_left_zz" => layer_3_left_zz,
		"layer_3_right_zz" => layer_3_right_zz,
		"layer_4_xzxz" => layer_4_xzxz,
		"layer_4_zxzx" => layer_4_zxzx
		)
end

function x_2_x_abef_meas()
	ancs = x_2_x_abef_ancs()
	tag = "x_2_x_abef"
	circs = measurement_subcircuits(tag)

	layer_1 = vcat(
		vcat(map(circs["zzzz_1"], ancs["layer_1_zzzz"])...),
		vcat(map(circs["zzzz_row_1"], ancs["layer_1_zzzz_row"])...),
		vcat(map(circs["xxxx_1"], ancs["layer_1_xxxx"])...)
		)

	layer_2 = vcat(
		vcat(map(circs["down_zz_1"], ancs["layer_2_down_zz"])...),
		vcat(map(circs["down_xx_1"], ancs["layer_2_down_xx"])...),
		vcat(map(circs["zxzx_col_1"], ancs["layer_2_zxzx_col"])...),
		vcat(map(circs["zzzz_1"], ancs["layer_2_zzzz"])...)
		)

	layer_3 = vcat(
		vcat(map(circs["up_xx_1"], ancs["layer_3_up_xx"])...),
		vcat(map(circs["left_xx_1"], ancs["layer_3_left_xx"])...),
		vcat(map(circs["right_xx_1"], ancs["layer_3_right_xx"])...),
		vcat(map(circs["left_zz_1"], ancs["layer_3_left_zz"])...),
		vcat(map(circs["right_zz_1"], ancs["layer_3_right_zz"])...)
		)

	layer_4 = vcat(
		vcat(map(circs["xzxz_1"], ancs["layer_4_xzxz"])...),
		vcat(map(circs["zxzx_1"], ancs["layer_4_zxzx"])...)
		)

	layer_5 = vcat(
		vcat(map(circs["zzzz_2"], ancs["layer_1_zzzz"])...),
		vcat(map(circs["zzzz_row_2"], ancs["layer_1_zzzz_row"])...),
		vcat(map(circs["xxxx_2"], ancs["layer_1_xxxx"])...)
		)

	layer_6 = vcat(
		vcat(map(circs["down_zz_2"], ancs["layer_2_down_zz"])...),
		vcat(map(circs["down_xx_2"], ancs["layer_2_down_xx"])...),
		vcat(map(circs["zxzx_col_2"], ancs["layer_2_zxzx_col"])...),
		vcat(map(circs["zzzz_2"], ancs["layer_2_zzzz"])...)
		)

	layer_7 = vcat(
		vcat(map(circs["up_xx_2"], ancs["layer_3_up_xx"])...),
		vcat(map(circs["left_xx_2"], ancs["layer_3_left_xx"])...),
		vcat(map(circs["right_xx_2"], ancs["layer_3_right_xx"])...),
		vcat(map(circs["left_zz_2"], ancs["layer_3_left_zz"])...),
		vcat(map(circs["right_zz_2"], ancs["layer_3_right_zz"])...)
		)

	layer_8 = vcat(
		vcat(map(circs["xzxz_2"], ancs["layer_4_xzxz"])...),
		vcat(map(circs["zxzx_2"], ancs["layer_4_zxzx"])...)
		)

	vcat(layer_1, layer_2, layer_3, layer_4,
			layer_5, layer_6, layer_7, layer_8)
end

function x_2_x_abef_stabs()
	ancs = x_2_x_abef_ancs()
	
	zzzz = map(zzzz_stab, ancs["layer_1_zzzz"])
	xxxx = map(xxxx_stab, ancs["layer_1_xxxx"])
	zzzz = vcat(zzzz, map(zzzz_stab, ancs["layer_1_zzzz_row"]))
	
	down_zz = map(down_zz_stab, ancs["layer_2_down_zz"])
	down_xx = map(down_xx_stab, ancs["layer_2_down_xx"])
	zxzx = map(zxzx_stab, ancs["layer_2_zxzx_col"])
	zzzz = vcat(zzzz, map(zzzz_stab, ancs["layer_2_zzzz"]))

	up_xx = map(up_xx_stab, ancs["layer_3_up_xx"])
	left_xx = map(left_xx_stab, ancs["layer_3_left_xx"])
	right_xx = map(right_xx_stab, ancs["layer_3_right_xx"])
	
	xzxz = map(xzxz_stab, ancs["layer_4_xzxz"])
	zxzx = vcat(zxzx, map(zxzx_stab, ancs["layer_4_zxzx"]))

	vcat(zzzz, xxxx, down_zz, down_xx,
			zxzx, up_xx, left_xx, right_xx, xzxz)
end

function s_x_meas()
	centers = [(1, 9), (1, 13), (1, 17), (1, 21),
				(9, 9), (9, 13), (9, 17), (9, 21)]

	function single_s_x_meas(anc)

		function data_meas(q)
			meas_name = measurement_name("s_x_meas", "x", q, 3)
			Measurement(meas_name, x_on(q))
		end

		[Preparation(x_on(anc)), cnot(anc, anc .+ (-1, 1)),
			cnot(anc .+ (1, 1), anc), cnot(anc, anc .+ (-1, 1)),
			s(anc .+ (-1, 1)), data_meas((anc .+ (1, 1))), cnot(anc .+ (1, 1), anc),
			data_meas((anc .+ (1, -1))), data_meas((anc .+ (-1, 1))),
			data_meas((anc .+ (-1, -1))), data_meas((anc .+ (1, 1)))]
	end

	vcat(map(single_s_x_meas, centers)...)
end

# ------------------------ Postprocessing --------------------------- #

"""
For each computational state that gets output, we need to:

	+ convert physical measurement results into logical ones
	+ decide whether the run is post-selected out:
 		- If an odd number of SX measurement results have returned a -1
		 	outcome, or
		- If perfect error detection on the three logical output qubits
			would show an error.
	+ perform logical corrections
	
"""
# function postprocess(circuit, comp_state::ComputationalState)
# 	log_results = circuit.log_meas_res(comp_state) 
# 	ps = circuit.postselect(comp_state, log_results)
	
# 	if ~ps
# 		comp_state = circuit.log_corrections(comp_state, log_results)
# 	end

# 	comp_state, ps
# end

function logical_measurement_results(comp_state::ComputationalState)
	phys_results = comp_state.meas_output

	parity(names) = reduce(xor, map(name -> phys_results[name], names))
	name(tag, stab) = coord -> measurement_name(tag, stab, coord, 2)
	# measurement_name(tag, stab, anc, round)
	x_1_x_abcd_names = vcat(
		map(name("x_1_x_abcd", "zxzx"),
			[(3, 5), (3, 9), (3, 13), (3, 17), (3, 21)]),
		map(name("x_1_x_abcd", "zzzz"),
			[(5, 7), (5, 11), (5, 15), (5, 19)]),
		map(name("x_1_x_abcd", "zz"),
			[(5, 5), (7, 9), (7, 13), (7, 17), (5, 21)])
			)

	x_abcdefgh_names = vcat(
		map(name("x_abcdefgh", "zxzx"),
			[(3, 9), (3, 13), (3, 17), (3, 21)]),
		map(name("x_abcdefgh", "xzxz"), 
			[(7, 9), (7, 13), (7, 17), (7, 21)]),
		map(name("x_abcdefgh", "zzzz"), 
			[(5, 11), (5, 15), (5, 19)]),
		map(name("x_abcdefgh", "zz"), [(5, 7), (5, 23)])
		)
	
	x_3_x_aceg_names = vcat(
		map(name("x_3_x_aceg", "xzxz"), [(7, 5), (7, 13), (7, 21)]),
		map(name("x_3_x_aceg", "zxzx"), [(3, 13), (3, 21)]),
		map(name("x_3_x_aceg", "zzzz"),
			[(5, 7), (5, 11), (5, 15), (5, 19)]),
		measurement_name("x_3_x_aceg", "xxx", (5, 5), 2),
		map(name("x_3_x_aceg", "zz"),
			[(5, 5), (3, 9), (3, 17), (5, 23), (7, 9), (7, 17)])
		)

	x_2_x_abef_names = vcat(
		map(name("x_2_x_abef", "zxzx"), [(3, 17), (3, 21)]),
		map(name("x_2_x_abef", "xzxz"), [(7, 17), (7, 21)]),
		measurement_name("x_2_x_abef", "zxzx_col", (5, 3), 2),
		map(name("x_2_x_abef", "zzzz"),
			[(5, 7), (5, 11), (5, 15), (5, 19)]),
		map(name("x_2_x_abef", "zz"), 
			[(3, 5), (3, 9), (3, 13), (7, 5), (7, 9), (7, 13), (5, 23)])
		)

	names = Dict("x_1_x_abcd" => x_1_x_abcd_names,
					"x_abcdefgh" => x_abcdefgh_names,
					"x_3_x_aceg" => x_3_x_aceg_names,
					"x_2_x_abef" => x_2_x_abef_names)

	ancs = [(1, 21), (1, 17), (1, 13), (1, 9),
			(9, 21), (9, 17), (9, 13), (9, 9)]
	log_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
	meas_shifts = [(-1, 1), (-1, -1), (1, -1)]
	name_3 = coord -> measurement_name("s_x_meas", "x", coord, 3)
	for (anc, log_name) in zip(ancs, log_names) 
		meas_coords = map(shift -> anc .+ shift, meas_shifts)
		key = string("s_x_meas_x_", log_name)
		names[key] = map(name_3, meas_coords)
	end

	Dict(key => parity(qubits) for (key, qubits) in names)
end

"""
`is_postselected(comp_state::ComputationalState, log_results)`

If any stabilizer measurement has returned a 1, the result will be
postselected.

Also, if the logical outputs have odd parity, we postselect.
"""
function is_postselected(comp_state::ComputationalState, log_results)
	meas_output = comp_state.meas_output

	parity(nms) = reduce(xor, map(nm -> meas_output[nm], nms))

	function nms(tag, stab)
		coord -> [measurement_name(tag, stab, coord, rnd)
					for rnd in 1:2]
	end
	
	tag = "x_1_x_abcd"
	x_1_x_abcd_names = vcat(
		vcat(map(nms(tag, "xxxx"), [(5, 9), (5, 13), (5, 17)])...),
		vcat(map(nms(tag, "xxx"), [(5, 5), (5, 21)])...),
		vcat(map(nms(tag, "down_xx"), [(1, 7), (1, 11), (1, 15), (1, 19), (1, 23)])...),
		vcat(map(nms(tag, "up_xx"), [(1, 3), (1, 7), (1, 11), (1, 15), (1, 19)])...),
		vcat(map(nms(tag, "zzzz"), [(1, 5), (1, 9), (1, 13), (1, 17), (1, 21)])...)
		)
		
	tag = "x_abcdefgh"
	x_abcdefgh_names = vcat(
		vcat(map(nms(tag, "xxxx"), [(5, 9), (5, 13), (5, 17), (5, 21)])...),
		vcat(map(nms(tag, "down_xx"), [(1, 11), (1, 15), (1, 19), (1, 23), (9, 11), (9, 15), (9, 19), (9, 23)])...),
		vcat(map(nms(tag, "up_xx"), [(1, 7), (1, 11), (1, 15), (1, 19), (9, 7), (9, 11), (9, 15), (9, 19)])...),
		vcat(map(nms(tag, "zzzz"), [(1, 9), (1, 13), (1, 17), (1, 21), (9, 9), (9, 13), (9, 17), (9, 21)])...)
		)

	tag = "x_3_x_aceg"
	x_3_x_aceg_names = vcat(
		vcat(map(nms(tag, "xxxx"), [(5, 9), (5, 13), (5, 17), (5, 21)])...),
		vcat(map(nms(tag, "xxx"), [(5, 5)])...),
		vcat(map(nms(tag, "up_xx"), [(1, 7), (1, 11), (1, 15), (1, 19),
			(9, 3), (9, 7), (9, 11), (9, 15), (9, 19)])...),
		vcat(map(nms(tag, "down_xx"), [(1, 11), (1, 15), (1, 19), (1, 23),
			(9, 7), (9, 11), (9, 15), (9, 19), (9, 23)])...),
		vcat(map(nms(tag, "zzzz"), [(1, 9), (1, 13), (1, 17), (1, 21),
			(9, 5), (9, 9), (9, 13), (9, 17), (9, 21)])...)
		)

	tag = "x_2_x_abef"
	x_2_x_abef_names = vcat(
		vcat(map(nms(tag, "xxxx"), [(5, 5), (5, 9), (5, 13), (5, 17), (5, 21)])...),
		vcat(map(nms(tag, "down_xx"), [(1, 11), (1, 15), (1, 19), (1, 23),
									(9, 11), (9, 15), (9, 19), (9, 23)])...),
		vcat(map(nms(tag, "up_xx"), [(1, 7), (1, 11), (1, 15), (1, 19),
									(9, 7), (9, 11), (9, 15), (9, 19)])...),
		vcat(map(nms(tag, "zzzz"), [(1, 9), (1, 13), (1, 17), (1, 21),
									(9, 9), (9, 13), (9, 17), (9, 21)])...)
		)

	data_name = coord -> measurement_name("s_x_meas", "x", coord, 3)
	centers = [(1, 9), (1, 13), (1, 17), (1, 21),
					(9, 9), (9, 13), (9, 17), (9, 21)]
	
	# weight_one_stab_names = map(data_name,
	# 							[ctr .+ (1, 1) for ctr in centers])

	weight_two_stab_sets = [map(data_name, 
								[ctr .+ (1, -1), ctr .+ (-1, -1)])
												for ctr in centers]

	log_result_keys = vcat("x_abcdefgh", [string("s_x_meas_x_", ltr)
										for ltr in collect('a' : 'h')])

	postselect_parity = Bool(mod(sum([log_results[key]
								for key in log_result_keys]), 2))

	nd_stab_names = vcat(x_1_x_abcd_names, x_abcdefgh_names, x_3_x_aceg_names, x_2_x_abef_names)
	error_detected = any(map(key -> meas_output[key] == 0x01, nd_stab_names))
	error_detected = error_detected || any(map(st -> mod(meas_output[st[1]] + meas_output[st[2]], 2) == 0x01,
								weight_two_stab_sets))

	error_detected || postselect_parity 
end

log_centers() = [(1, 5), (5, 1), (9, 5)]

function logical_operators()
	ctrs = log_centers()
	top(ctr) = [ctr .+ (1, 1), ctr .+ (-1, 1)]
	right(ctr) = [ctr .+ (1, 1), ctr .+ (1, -1)]
	z_on_set(st) = prod(map(z_on, st))
	x_on_set(st) = prod(map(x_on, st))
	
	Z = [z_on_set(top(ctrs[1])),
			z_on_set(right(ctrs[2])),
			z_on_set(top(ctrs[3]))]
	X = [x_on_set(right(ctrs[1])),
			x_on_set(top(ctrs[2])),
			x_on_set(right(ctrs[3]))]
	Z, X
end

"""
`output_stabilizers()`

In order to tell whether we're dealing with a logical error, we need to
know whether an output Pauli commutes with the stabilizer generators
and anticommutes with at least one logical.
"""
function output_stabilizers()
	x_on_set(st) = prod(map(x_on, st))
	z_on_set(st) = prod(map(z_on, st))

	ctrs = log_centers()
	
	s_z = map(ctr -> z_on_set([ctr .+ (1, 1), ctr .+ (-1, 1),
							ctr .+ (-1, -1), ctr .+ (1, -1)]), ctrs)
	
	s_x_1 = [x_on_set([ctrs[1] .+ (1, 1), ctrs[1] .+ (-1, 1)]), 
				x_on_set([ctrs[1] .+ (1, -1), ctrs[1] .+ (-1, -1)])]
	
	# one of the qubits in the diagram is sideways fyi
	s_x_2 = [x_on_set([ctrs[2] .+ (1, 1), ctrs[2] .+ (1, -1)]), 
				x_on_set([ctrs[2] .+ (-1, 1), ctrs[2] .+ (-1, -1)])]

	s_x_3 = [x_on_set([ctrs[3] .+ (1, 1), ctrs[3] .+ (-1, 1)]), 
				x_on_set([ctrs[3] .+ (1, -1), ctrs[3] .+ (-1, -1)])]
	
	vcat(s_z[1], s_x_1, s_z[2], s_x_2, s_z[3], s_x_3)
end

function do_logical_corrections(comp_state::ComputationalState, log_results)
	Z, X = logical_operators()

	# s_x_meas_x_h has no effect on logical corrections, only
	# post-selection
	log_keys = ["s_x_meas_x_a", "s_x_meas_x_b", "s_x_meas_x_c",
				"s_x_meas_x_d", "s_x_meas_x_e", "s_x_meas_x_f",
				"s_x_meas_x_g",
				"x_1_x_abcd", "x_2_x_abef", "x_3_x_aceg"]
	
	log_ops = [Z[1] * Z[2] * Z[3], Z[1] * Z[2], Z[1] * Z[3],
				Z[1], Z[2] * Z[3], Z[2],
				Z[3], 
				Z[1], Z[2], Z[3]]
	
	new_pauli = deepcopy(comp_state.pauli)

	for (key, log) in zip(log_keys, log_ops)
		if log_results[key] == 0x01
			new_pauli *= log
		end
	end

	# Note: a quick check in Quirk reveals that this correction is not
	# necessary, because we're preparing the state |+++>, using
	# transversal S instead of T
	# new_pauli *= X[1] * X[2] * X[3]

	ComputationalState(new_pauli, comp_state.meas_output)
end

# ----------------------- Circuit Analysis -------------------------- #

"""
Iterator over circuits with a single fault.
"""
function faulty_circuits(circuit)
	faults = vcat(input_faults(circuit),
					gate_faults(circuit))
	
	map(fault -> add_fault(circuit, fault), faults)
end

"""
Iterator over circuits with a single fault only on a gate (gets around
having to figure out which qubits in a circuit start as open wires).
"""
function circuits_with_faulty_gates(circuit)
	map(fault -> add_fault(circuit, fault), gate_faults(circuit))
end

function paulis_on(nq, qubit::Int64)
	[QC.single_z(nq, qubit),
	QC.single_x(nq, qubit),
	QC.single_y(nq, qubit)]
end

function phase_free_prod(ps)
	initial_prod = reduce(*, ps)
	QC.PauliOperator(0x00, initial_prod.nqubits, initial_prod.xz)
end

function paulis_on(nq, qs)
	id = identity_pauli(nq)
	small_nq = length(qs)
	single_paulis = map(q -> vcat([id], paulis_on(nq, q)), qs)
	
	big_paulis = map(phase_free_prod, Iterators.product(collect(repeat(single_paulis, small_nq))...))
	setdiff(reduce(vcat, big_paulis), [id])
end

function input_faults(circuit::Circuit)
	ps(qubit) = paulis_on(nq, qubit)
	fs(qubit) = map(p -> Fault(p, 1, 0), ps(qubit))
	vcat(map(fs, qubits(circuit))...)
end

function gate_faults(circuit)
	reduce(vcat, map(fault_set, enumerate(layers(circuit))))
end

function fault_set(dx_layer::Tuple{Int, Layer})
	layer_dx, layer = dx_layer
	glist = gatelist(layer)

	f_set(gate_dx) = fault_set((layer_dx, gate_dx, glist[gate_dx]))

	reduce(vcat, map(f_set, 1:length(glist)))
end

"""
only one kind of fault is possible after a preparation
"""
function flip_pauli(prep_meas::Union{Preparation, Measurement})
	q = qubits(prep_meas)[1]
	if prep_meas.pauli[q] == (false, true)
		pauli = QC.single_x(nq, q)
	elseif prep_meas.pauli[q] == (true, false)
		pauli = QC.single_z(nq, q)
	else 
		pauli = QC.single_x(nq, q)
	end

	return pauli
end

function fault_set(dx_dx_meas::Tuple{Int, Int, Measurement})
	layer_dx, gate_dx, meas = dx_dx_meas
	[Fault(flip_pauli(meas), layer_dx, gate_dx-1)]
end

function fault_set(dx_dx_prep::Tuple{Int, Int, Preparation})
	layer_dx, gate_dx, prep = dx_dx_prep
	[Fault(flip_pauli(prep), layer_dx, gate_dx)]
end

function fault_set(dx_dx_gate::Tuple{Int, Int, Any})
	layer_dx, gate_dx, gate = dx_dx_gate
	map(pauli -> Fault(pauli, layer_dx, gate_dx),
						paulis_on(nq, qubits(gate)))
end

function is_logical(circuit, pauli)
	stabs = circuit.final_stabs
	logs = vcat(circuit.log_ops...)

	# if it could be detected by a subsequent perfect round, it's not a
	# logical error.
	for stab in stabs
		if QC.comm(stab, pauli) == 0x01
			return false
		end
	end

	for log in logs 
		if QC.comm(log, pauli) == 0x01
			return true
		end
	end

	false
end

function contains_logical_error(circuit, state)
	results = circuit.log_meas_res(state)
	state = circuit.log_corrections(state, results)
	~circuit.postselect(state, results) && is_logical(circuit, state.pauli)
end

"""
`pathological(f_circ)`
If a faulty circuit results in a logical Pauli and not postselection,
it's not fault-tolerant to distance 2.
"""
function pathological(f_circ)
	state = run(f_circ)
	contains_logical_error(f_circ, state)
end

"""
For testing purposes, determines the entire set of O(p) faults that
cause logical failure.
"""
function pathological_single_faults(circuit)
	faulty_circs = faulty_circuits(circuit)
	singles = []
	for f_circ in faulty_circs
		if pathological(f_circ)
			push!(singles, f_circ)
		end
	end
	singles
end
"""
Determines whether any single gate fault will be post-selected out at
some point.
"""
function is_d_2_fault_tolerant(circuit)
	faulty_circs = faulty_circuits(circuit)
	for f_circ in faulty_circs
		if pathological(f_circ)
			return false
		end
	end
	true
end

function combine(state_1::ComputationalState, state_2::ComputationalState)
	new_pauli = deepcopy(state_1.pauli) * deepcopy(state_2.pauli)
	
	new_output = deepcopy(state_1.meas_output)
	output_2 = state_2.meas_output
	for key in keys(new_output)
		new_output[key] = xor(new_output[key], output_2[key])
	end

	ComputationalState(new_pauli, new_output)
end

"""
Obtains results (measurement record and Pauli) from runs of a circuit
with one fault, then takes pairs of those outputs and XORs the
measurement results and multiplies the Paulis to see whether the final
effect of both faults will be a logical that's not post-selected out.  

Note: As of 17 July 2023:
malicious_fault_pairs(lattice_surgery_circuit()) == 5926 
"""
function malicious_fault_pairs(circuit)
	faulty_circs = faulty_circuits(circuit)
	single_fault_states = map(run, faulty_circs)
	
	n_malicious_pairs = 0
	n_states = length(single_fault_states)
	# Threads.@threads for pair_1_dx in 1:n_states
	# 	pair_1 = single_fault_states[pair_1_dx]
	# 	for pair_2_dx in pair_1_dx : n_states
	# 		pair_2 = single_fault_states[pair_2_dx]
	# 		if contains_logical_error(circuit, combine(pair_1, pair_2))
	# 			n_malicious_pairs += 1
	# 		end
	# 	end
	# end
	PM.@showprogress for pair in IT.subsets(single_fault_states, 2)
		if contains_logical_error(circuit, combine(pair[1], pair[2]))
			n_malicious_pairs += 1
		end
	end

	n_malicious_pairs
end

"""
Same as `malicious_fault_pairs`, but doesn't include input faults, for
simplicity.
"""
function malicious_gate_fault_pairs(circuit)
	single_fault_states = map(run, circuits_with_faulty_gates(circuit))
	
	n_malicious_pairs = 0
	PM.@showprogress for pair in IT.subsets(single_fault_states, 2)
		if contains_logical_error(circuit, combine(pair[1], pair[2]))
			n_malicious_pairs += 1
		end
	end

	n_malicious_pairs
end

end # module
