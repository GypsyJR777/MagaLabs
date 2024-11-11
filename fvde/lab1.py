import json
import sys

def getIndecesFromTable(num, table):
    index = int(num, 2)
    return table[index]

def getIndesFromTable(num1, num2):
    if (num1 == '0' and num2 == '0'):
        return 0
    elif (num1 == '0' and num2 == '1'):
        return 1
    elif (num1 == '1' and num2 == '0'):
        return 2
    elif (num1 == '1' and num2 == '1'):
        return 3
    else: raise RuntimeError("False nums")

class LogicGate:
    def __init__(self, inw, outw, table, all):
        self.inw = inw
        self.outw = outw
        self.table = table
        self.all = all

    def getOutw(self):
        return self.outw

    def compute(self, inputs):
        if len(inputs) != self.inw:
            raise ValueError(f"Expected {self.inw} inputs, got {len(inputs)}. Inputs: {inputs}")

        nums = [format(num, 'b').zfill(self.all) for num in inputs]
        
        result = ''

        for j in range(len(nums[0])):
            num = ''
            for i in range(self.inw):
                num += nums[i][j]
            result += str(getIndecesFromTable(num, self.table))
        
        # for i in range(1, len(nums)):
        #     k = ''
        #     for j in range(self.all):
        #         index = getIndesFromTable(result[j], nums[i][j])
        #         k += str(self.table[index])
        #     result = k

        return int(result, 2)

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
                        outputs[f"{gate_name}{i}"] = outs
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
        input_values = [int(line.strip(), 16) for line in values_file]

    results = compute_outputs(gates, schematic, input_values)

    for _, result in results.items():
        print(f"0x{int(result):X}")

if __name__ == "__main__":
    main()
