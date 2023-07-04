module QubitLayout
import IterTools as IT

export addrs, all_qubits, nq

const data_xs = collect(0:2:10)
const ancilla_xs = collect(1:2:9)
const data_ys = collect(0:2:22)
const ancilla_ys = collect(1:2:23)
const data_qubits = collect(vcat(IT.product(data_xs, data_ys)...))
const ancilla_qubits = collect(vcat(IT.product(ancilla_xs, ancilla_ys)...))
const all_qubits = vcat(data_qubits, ancilla_qubits)
const addrs = Dict(qubit => dx for (dx, qubit) in enumerate(all_qubits))
const nq = length(all_qubits)

end