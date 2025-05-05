import numpy as np
import os

input_file = "fct_astra_sim.txt"
output_dir = "data/astra_sim/ns3/output"
os.makedirs(output_dir, exist_ok=True)

fsize = []
fat = []
fsd = []

host_id_map = {}
host_counter = 0

with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # skip incomplete lines

        sip_raw = parts[0]
        dip_raw = parts[1]
        size = int(parts[4])  # column 5: size (bytes)
        start_time = float(parts[5])  # column 6: start time (sec)

        # Map to consecutive integers
        for host in [sip_raw, dip_raw]:
            if host not in host_id_map:
                host_id_map[host] = host_counter
                host_counter += 1

        sip = host_id_map[sip_raw]
        dip = host_id_map[dip_raw]

        fsd.append([sip, dip])
        fsize.append(size)
        fat.append(int(start_time * 1e9))  # convert sec to ns

# Save
np.save(os.path.join(output_dir, "fsize.npy"), np.array(fsize, dtype=np.int64))
np.save(os.path.join(output_dir, "fat.npy"), np.array(fat, dtype=np.int64))
np.save(os.path.join(output_dir, "fsd.npy"), np.array(fsd, dtype=np.int32))

print(f"Saved to: {output_dir}")
print(f"Flows: {len(fsize)} | Unique hosts: {len(host_id_map)}")
