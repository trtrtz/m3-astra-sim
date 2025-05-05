import numpy as np

input_file = "fct_astra_sim.txt"
output_dir = "data/astra_sim/ns3/output"  # 和fsize.npy etc. 一致

fct_list = []

with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        fct_ns = float(parts[6])  # Column 7 = fct (ns)
        fct_list.append(fct_ns)

fct_array = np.array(fct_list, dtype=np.float64)
np.save(f"{output_dir}/fct_flowsim.npy", fct_array)

print(f"Saved fct_flowsim.npy with {len(fct_array)} flows to {output_dir}")
