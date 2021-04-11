import cirq
import numpy as np

def controlled_sqrt_iswap(qs):
    c = cirq.Circuit()
    c.append(cirq.ControlledGate(cirq.Y ** -0.5).on(qs[0], qs[2]))
    c.append(cirq.CCZ.on(qs[0], qs[1], qs[2]))
    c.append(cirq.ControlledGate(cirq.Y ** 0.5).on(qs[0], qs[2]))
    c.append(cirq.ControlledGate(cirq.Y ** 0.5).on(qs[0], qs[1]))
    c.append(cirq.CX.on(qs[0], qs[1]))
    c.append(cirq.ControlledGate(cirq.Y ** -0.5).on(qs[0], qs[1]))
    c.append(cirq.CCZ(qs[0], qs[1], qs[2]))
    c.append(cirq.ControlledGate(cirq.Y**0.5).on(qs[0], qs[1]))
    c.append(cirq.ControlledGate(cirq.T).on(qs[0], qs[1]))
    c.append(cirq.ControlledGate(cirq.Y**-0.5).on(qs[0], qs[1]))
    c.append(cirq.CCZ(qs[0], qs[1], qs[2]))
    c.append(cirq.ControlledGate(cirq.Y**0.5).on(qs[0], qs[1]))
    c.append(cirq.ControlledGate(cirq.T ** -1).on(qs[0], qs[1]))
    c.append(cirq.ControlledGate(cirq.Y**0.5).on(qs[0], qs[1]))
    c.append(cirq.CX.on(qs[0], qs[1]))
    c.append(cirq.ControlledGate(cirq.Y ** -0.5).on(qs[0], qs[2]))
    c.append(cirq.CCZ.on(qs[0], qs[1], qs[2]))
    c.append(cirq.ControlledGate(cirq.Y ** 0.5).on(qs[0], qs[2]))
    return c

if __name__=="__main__":
    c = controlled_sqrt_iswap(cirq.LineQubit.range(3))
    print(c)

    print()
    print()
    print()

    g = cirq.ControlledGate(cirq.ISWAP ** 0.5)
    g = cirq.decompose(g.on(*cirq.LineQubit.range(3)))
    c = cirq.Circuit(g)
    print(c)

    g = cirq.ControlledGate(cirq.FSimGate(-np.pi/4, 0))
    g = cirq.decompose(g.on(*cirq.LineQubit.range(3)))
    c = cirq.Circuit(g)
    print(c)
