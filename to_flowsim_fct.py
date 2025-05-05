import os

#input and output file paths
input_fct_file = "/users/tingzhou/m3/fct_astra_sim.txt"
output_fct_file = "/users/tingzhou/m3/fct.txt"


with open(input_fct_file, "r") as f:
    lines = f.readlines()

num_flows = len(lines)  #num of flows

#add id
with open(output_fct_file, "w") as f:
    #write the number of flows as the first line
    f.write(f"{num_flows}\n")

    for flow_id, line in enumerate(lines):
        data = line.strip().split()

        #extract fields of interest
        src_ip, dst_ip, src_port, dst_port = data[:4]
        size = int(data[4])          # Flow size in Bytes
        start_time = int(float(data[5]) * 1e9)  
        fct = int(float(data[6]))     # FCT in ns
        standalone_fct = int(float(data[7]))  # Standalone FCT in ns
        
        #Convert src_ip and dst_ip to numerical values if needed
        src = hash(src_ip) % (10**6)  #hash to a numeric ID 
        dst = hash(dst_ip) % (10**6)  #same for dst

        #format as required by m3 files
        f.write(f"{flow_id} {src} {dst} ? 100 {size} {start_time} {fct} {standalone_fct} ?\n")

print(f"Converted file saved as: {output_fct_file}")
