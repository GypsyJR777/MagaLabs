import json
import sys

class LogicGate:
    def __init__(self, inw, outw, table, all):
        self.inw = inw
        self.outw = outw
        self.table = table
        self.all = all

    def getOutw(self):
        return self.outw

    def getIndecesFromTable(self, num, k):
        index = int(num, 2)
        return (self.table[index] >> k) & 1

    def compute(self, inputs):
        if len(inputs) != self.inw:
            raise ValueError(f"Expected {self.inw} inputs, got {len(inputs)}. Inputs: {inputs}")

        nums = [format(num, 'b').zfill(self.all) for num in inputs]
        
        results = []

        for k in range(self.outw):
            result = ''
            for j in range(len(nums[0])):
                num = ''
                for i in range(self.inw):
                    num += nums[i][j]
                result += str(self.getIndecesFromTable(num, k))
            results.append(int(result, 2))
        
        return results

def load_gates(gates_json, all):
    gates = {}
    for name, params in gates_json.items():
        gates[name] = LogicGate(params['inw'], params['outw'], params['table'], all)
    return gates

def load_schematic(schematic_json):
    return {
        'inw': schematic_json['inw'],
        'outw': schematic_json['outw'],
        'gates': schematic_json['gates'],
        'drivers': schematic_json['drivers'],
        'outputs': schematic_json['output']
    }

def compute_outputs(gates, schematic, inputs):
    results = {}
    signals = {}
    for key, value in schematic["drivers"].items():
        if isinstance(value, int):
            signals[key] = inputs[int(value)]

    # Debug: Show initial input signals
    # print("Initial signals:", signals)

    outputs = {}
    while (len(results) < schematic["outw"]):
        for gate_name, gate_type in schematic['gates'].items():
            gate = gates[gate_type]

            gate_inputs = []
            if (f"{gate_name}0" not in outputs):
                for i in range(gate.inw):
                    driver_key = f"{gate_name}{i}"
                    if driver_key in signals:
                        gate_inputs.append(signals[driver_key])
                    elif schematic['drivers'][driver_key] in outputs:
                        gate_inputs.append(outputs[schematic['drivers'][driver_key]])

                if (len(gate_inputs) == gate.inw):
                    outs = gate.compute(gate_inputs)
                    for i in range(gate.getOutw()):
                        outputs[f"{gate_name}{i}"] = outs[i]
                # Debug: Show computed output for this gate
                # print(f"Computed {gate_name}: {schematic['drivers'][f"{gate_name}{i}"]} with inputs {gate_inputs}")

        for key in schematic['outputs']:
            if (key in outputs and key not in results):
                results[key] = outputs[key]

    return results


def main():
    if len(sys.argv) != 3:
        print("Usage: <cmd> <in.json> <values.txt>")
        return


    with open(sys.argv[1], 'r') as json_file:
        circuit_json = json.load(json_file)

    schematic = load_schematic(circuit_json['schematics'])
    gates = load_gates(circuit_json['gates'], schematic["inw"])

    with open(sys.argv[2], 'r') as values_file:
        input_values = [
            int(line.strip(), 2**schematic["inw"]) for line in values_file
        ]

    results = compute_outputs(gates, schematic, input_values)

    for _, result in results.items():
        print(f"0x{int(result):X}")

if __name__ == "__main__":
    main()
