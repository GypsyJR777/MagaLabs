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

        nums = [format(num, 'b') for num in inputs]
        
        results = []

        for k in range(self.outw):
            result = ''
            num = ''
            for i in range(self.inw):
                num += nums[i]
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
            signals[key] = int(inputs[int(value)])

    # Debug: Show initial input signals
    # print("Initial signals:", signals)

    outputs = {}
    while (len(results) < len(set(schematic['outputs']))):
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

        for key in schematic['outputs']:
            if (key in outputs and key not in results):
                results[key] = outputs[key]
            elif (isinstance(key, int) and key not in results):
                results[key] = int(inputs[int(key)])

    # print(outputs)
    return results


def main():
    if len(sys.argv) != 3:
        print("Usage: <cmd> <in.json> <values.txt>")
        return


    with open(sys.argv[1], 'r') as json_file:
        circuit_json = json.load(json_file)

    schematic = load_schematic(circuit_json['schematics'])
    gates = load_gates(circuit_json['gates'], schematic["inw"])

    input_values_from_file = []

    with open(sys.argv[2], 'r') as values_file:
        for line in values_file:
            line.strip()
            if ("0x" in line):
                num = int(line.replace("0x", ""), 16)
            else:
                num = int(line)
            
            if(num > 2**schematic["inw"] - 1):
                raise ValueError(f"Value {num} is out of range for {schematic['inw']} bits")
                
            input_values_from_file.append(num)

    input_values = []
    for value in input_values_from_file:
        input_values.append(str(format(value, 'b').zfill(schematic["inw"]))[::-1])

    results = []
    for value in input_values:
        results.append(compute_outputs(gates, schematic, list(value)))

    # print(results)
    for result in results:
        res = ""
        for key in schematic['outputs']:
            res = str(result[key]) + res
            # print(f"{key} = {result[key]} : {res}")
        # print(f"{res} = 0x{int(res, 2):X}")
        print(f"0x{int(res, 2):X}")

if __name__ == "__main__":
    main()
