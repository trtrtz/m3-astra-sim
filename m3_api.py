import numpy as np
import yaml
import sys
import os 

sys.path.insert(0, "./util")
from util.consts import (
    P99_PERCENTILE_LIST,
    PERCENTILE_METHOD,
    MTU,
    HEADER_SIZE,
    BYTE_TO_BIT,
    BDP_DICT,
    LINK_TO_DELAY_DICT,
    UNIT_G,
    get_base_delay_pmn,
    get_size_bucket_list,
    get_size_bucket_list_output,
    EPS
)
from util.utils import (
    fix_seed,
)
from util.model import FlowSimTransformer_Path
import torch
from util.arg_parser import create_config
import json
PARAM_VEC_INIT=np.array([0,30,18,1,1,0,0,0,30,0,0,0,0,0,0])

args = create_config()
fix_seed(args.shard)

#DEVICE = torch.device(args.device) 
#DEVICE = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set parameters
model_trained_dir=f"/users/tingzhou/m3/astra_sim_api/version_1"
#output_dir=f"./ckpts"
#model_id=""

def extract_flows_from_fct(fct_file_path):
    """Reads fct.txt and extracts flows as (src, dst, size, start_time, fct)."""
    flows = []
    with open(fct_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:  # Ensure valid flow data
                continue
            src = int(parts[2])   # sid 
            dst = int(parts[3])   # did
            size = int(parts[4])  # size (Bytes)
            start_time = int(parts[5])  # Start time (ns)
            fct = int(parts[6])  # Flow completion time (ns)
            flows.append([src, dst, size, start_time, fct])
    return flows

class m3_inference:
    def __init__(self,config_path):
        self.bucket_thold = 1
        
        # load data list
        self.dir_train = model_trained_dir
        f = open(f"{self.dir_train}/data_list.json", "r")
        data_list=json.loads(f.read())
        # [["shard1191_nflows20000_nhosts7_lr10Gbpsparam_k30", [0, 6], "_topo-pl-7_dctcp"],...]
        self.data_list = data_list["test"]
        # load config
        with open(f'{config_path}', "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            dataset_config = config["dataset"]
            model_config = config["model"]
            training_config = config["training"]
            others = config["others"]
        # 仿真数据数据输入
        self.dir_data_input = others["dir_data_input"]
        # 训练模型路径
        self.dir_train = others["dir_train"]
        # 最后结果输出路径
        self.output_dir = others["output_dir"]
        # 模型id
        self.model_id = others["model_id"]
        n_params=dataset_config["n_params"]
        model = FlowSimTransformer_Path.load_from_checkpoint(
            f"{self.dir_train}/checkpoints/last{self.model_id}.ckpt",
            #map_location=DEVICE,=“
            #map_location=torch.device('cuda:0'),
            map_location = torch.device('cpu'),
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            n_embd=model_config["n_embd"],
            block_size=model_config["block_size"],
            vocab_size=model_config["vocab_size"],
            dropout=model_config["dropout"],
            compile=model_config["compile"],
            loss_fn_type=model_config["loss_fn_type"],
            weight_decay=training_config["weight_decay"],
            learning_rate=training_config["learning_rate"],
            betas=training_config["betas"],
            batch_size=training_config["batch_size"],
            enable_val=training_config["enable_val"],
            enable_dist=training_config["enable_dist"],
            enable_masked_loss=training_config["enable_masked_loss"],
            enable_weighted_loss=training_config["enable_weighted_loss"],
            enable_context=dataset_config.get("enable_context", False),
            hidden_sizes=model_config["hidden_dims"],
            enable_position=model_config["enable_position"],
            enable_log=training_config["enable_log"],
            n_params=n_params,
            save_dir=self.output_dir,
        )
        
        model.eval()
        self.model=model
        self.lr=10
        self.enable_context = dataset_config.get("enable_context", False)
        self.enable_log=training_config["enable_log"]
        bdp_dict_db = {}
        bdp_dict_db_output = {}
        for n_hosts in [3,5,7]:
            BDP = 10 * MTU
            bdp_dict_db[n_hosts] = get_size_bucket_list(mtu=MTU, bdp=BDP)
            bdp_dict_db_output[n_hosts] = get_size_bucket_list_output(mtu=MTU, bdp=BDP)
        self.bdp_dict_db = bdp_dict_db
        self.bdp_dict_db_output = bdp_dict_db_output

        model.export_to_bin_llama_v0(filepath=f"{self.output_dir}/model_llama{self.model_id}.bin")
        model.export_to_bin_mlp(filepath=f"{self.output_dir}/model_mlp{self.model_id}.bin")
        
    def run_inference(self,idx):
        spec, src_dst_pair_target, topo_type = self.data_list[idx]
        n_hosts = int(spec.split("_")[2][6:])

        size_bucket_list = self.bdp_dict_db[n_hosts]
        size_bucket_list_output = self.bdp_dict_db_output[n_hosts]
        spec=f'data_lr10Gbps'
        
        param_data = PARAM_VEC_INIT
        if param_data[3]==1.0:
            param_data=np.insert(param_data,4,0)
        else:
            param_data=np.insert(param_data,4,1)
             
        param_data=np.insert(param_data,0,[0,0,0])
        param_data[n_hosts//2-1]=1.0
        param_data[3]=BDP_DICT[n_hosts]/MTU
        print(f"param_data: {param_data}")
        
        with open(f"{self.dir_data_input}/fct.txt", "r") as f:
            num_lines = int(f.readline().strip())  # Read the number of flows
            flow_src_dst = []
            sizes = []
            fcts = []

            for _ in range(num_lines):
                data = f.readline().strip().split()
                sip = int(data[2])  # Source ID
                dip = int(data[3])  # Destination ID
                size = int(data[4])  # Size in bytes
                fct = int(data[6])  # Flow completion time in nanoseconds
                
                flow_src_dst.append([sip, dip])
                sizes.append(size)
                fcts.append(fct)
        # fid = np.array(flow_id).astype("int32")
        sizes_flowsim = np.array(sizes).astype("int64")
        flow_src_dst_flowsim = np.array(flow_src_dst).astype("int32")

        sizes=sizes_flowsim
        flow_src_dst=flow_src_dst_flowsim
        
        flow_idx_target_flowsim = np.logical_and(
            flow_src_dst_flowsim[:, 0] == src_dst_pair_target[0],
            flow_src_dst_flowsim[:, 1] == src_dst_pair_target[1],
        )
        flow_idx_nontarget_flowsim=~flow_idx_target_flowsim
        flow_idx_nontarget_internal_flowsim=np.logical_and(
            flow_src_dst_flowsim[:, 0] != src_dst_pair_target[0],
            flow_src_dst_flowsim[:, 1] != src_dst_pair_target[1],
        )
        flow_src_dst_flowsim[:, 0] = np.clip(flow_src_dst_flowsim[:, 0], 0, len(LINK_TO_DELAY_DICT[n_hosts]) - 1)
        flow_src_dst_flowsim[:, 1] = np.clip(flow_src_dst_flowsim[:, 1], 0, len(LINK_TO_DELAY_DICT[n_hosts]) - 1)

        n_links_passed = abs(flow_src_dst_flowsim[:, 0] - flow_src_dst_flowsim[:, 1])+flow_idx_nontarget_flowsim+flow_idx_nontarget_internal_flowsim
        delay_comp=LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,0]]+LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,1]] 

        DELAY_PROPAGATION_PERFLOW = get_base_delay_pmn(
            sizes=sizes, n_links_passed=n_links_passed, lr_bottleneck=self.lr,flow_idx_target=flow_idx_target_flowsim,flow_idx_nontarget_internal=flow_idx_nontarget_internal_flowsim
        )+delay_comp
        fcts_flowsim = (
            np.load(f"{self.dir_data_input}/fct_flowsim.npy") + DELAY_PROPAGATION_PERFLOW
        )
        i_fcts_flowsim = (
            sizes + np.ceil(sizes / MTU) * HEADER_SIZE
        ) * BYTE_TO_BIT / self.lr + DELAY_PROPAGATION_PERFLOW
        sldns_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
        sldns_flowsim = np.clip(sldns_flowsim, a_max=None, a_min=1.0)
        
        sldns=sldns_flowsim
        sldns_list = []
        bins = []
        x_len = len(size_bucket_list) + 1
        y_len = len(P99_PERCENTILE_LIST)

        # add the target flow
        # print("size_bucket_list: ",size_bucket_list)
        sldns_flowsim_target = sldns_flowsim[flow_idx_target_flowsim]
        sldns_list.append(sldns_flowsim_target)
        bins_target = np.digitize(sizes_flowsim[flow_idx_target_flowsim], size_bucket_list)
        bins.append(bins_target)
        
        if self.enable_context:
            for link_idx_internal in range(
                src_dst_pair_target[0], src_dst_pair_target[1]
            ):
                flow_idx_selected = np.logical_and(
                    flow_src_dst_flowsim[:, 0] <= link_idx_internal,
                    flow_src_dst_flowsim[:, 1] > link_idx_internal,
                )
                flow_idx_selected = np.logical_and(flow_idx_selected, ~flow_idx_target_flowsim)
                sizes_perlink = sizes_flowsim[flow_idx_selected]
                sldns_flowsim_perlink = sldns_flowsim[flow_idx_selected]
                
                sldns_list.append(sldns_flowsim_perlink)
                bins.append(np.digitize(sizes_perlink, size_bucket_list))

        n_sldns_list = len(sldns_list)
        sizebucket_to_sldn = np.zeros((n_sldns_list, x_len, y_len))
        num_flows_per_cell = np.zeros((n_sldns_list, x_len, y_len))
        n_sizes_effective = np.ones((n_sldns_list, 1))

        for sldns_idx in range(n_sldns_list):
            if len(bins[sldns_idx]) == 0:
                continue
            for x_idx in range(x_len):
                sldn_idx_target = np.nonzero(bins[sldns_idx] == x_idx)[0]
                if len(sldn_idx_target) < self.bucket_thold:
                    continue
                
                sldns_tmp = sldns_list[sldns_idx][sldn_idx_target]
                sizebucket_to_sldn[sldns_idx, x_idx] = np.percentile(
                    sldns_tmp, P99_PERCENTILE_LIST,
                    method=PERCENTILE_METHOD
                )
                num_flows_per_cell[sldns_idx, x_idx] = len(sldn_idx_target)
                n_sizes_effective[sldns_idx] += len(sldn_idx_target)
        res = sizebucket_to_sldn.reshape((n_sldns_list, -1)).astype(np.float32)
        
        for i in range(len(sldns_list)):
            print(f"Sort Bucket {i}: {num_flows_per_cell[i,:,0]}")
            print(f"feat-input-{i}: {res[i,0]}, {res[i,-1]}")
        
        num_flows_per_cell = num_flows_per_cell.reshape((n_sldns_list, -1)).astype(
            np.float32
        )
        
        num_flows_per_cell = np.divide(num_flows_per_cell, n_sizes_effective)

        # find foreground/background flows for gt
        flow_idx_target = np.logical_and(
            flow_src_dst[:, 0] == src_dst_pair_target[0],
            flow_src_dst[:, 1] == src_dst_pair_target[1],
        )
        # output/ground truth
        sldns_output = sldns[flow_idx_target]
        bins_output = np.digitize(sizes[flow_idx_target], size_bucket_list_output)
        x_len_output = len(size_bucket_list_output) + 1
        sizebucket_to_sldn_output = np.ones((x_len_output, y_len))
        num_flows_per_cell_output = np.zeros((x_len_output, y_len))
        n_sizes_effective_output = 0

        for x_idx in range(x_len_output):
            sldn_idx_target = np.nonzero(bins_output == x_idx)[0]
            if len(sldn_idx_target) < self.bucket_thold:
                continue

            sldns_tmp = sldns_output[sldn_idx_target]
            sizebucket_to_sldn_output[x_idx] = np.percentile(
                sldns_tmp, P99_PERCENTILE_LIST,
                method=PERCENTILE_METHOD
            )
            num_flows_per_cell_output[x_idx] = len(sldn_idx_target)
            n_sizes_effective_output += len(sldn_idx_target)
        res_output = sizebucket_to_sldn_output.reshape((-1)).astype(np.float32)

        num_flows_per_cell_output = num_flows_per_cell_output.reshape((-1)).astype(
            np.float32
        )
        if n_sizes_effective_output == 0:
            num_flows_per_cell_output[:] = 0
        else:
            num_flows_per_cell_output = np.divide(
            num_flows_per_cell_output, n_sizes_effective_output
        )

        # [size_bucket,percentile]
        n_input = n_sldns_list
        # res -= 1.0
        assert (res>=0).all()
        res=np.insert(res, res.shape[1], param_data[:,None], axis=1)
        sizebucket_to_sldn_flowsim=torch.tensor(res).to(DEVICE)
        num_flows_per_cell_flowsim=torch.tensor(num_flows_per_cell).to(DEVICE)
        sizebucket_to_sldn=torch.tensor(res_output).to(DEVICE)
        num_flows_per_cell=torch.tensor(num_flows_per_cell_output)
        sizebucket_to_sldn_flowsim_idx=[n_input]
        src_dst_pair_target=np.array(src_dst_pair_target)
        spec=[spec]
        
        with torch.no_grad():
            if self.model.enable_const_opt:
                num_flows_per_cell_flowsim=num_flows_per_cell_flowsim.reshape((num_flows_per_cell_flowsim.shape[0],-1,self.model.y_len))
                num_flows_per_cell_flowsim=num_flows_per_cell_flowsim.mean(dim=-1)
                for idx_1 in range(num_flows_per_cell_flowsim.shape[0]):
                    for idx_2 in range(num_flows_per_cell_flowsim.shape[1]):
                        if num_flows_per_cell_flowsim[idx_1,idx_2]<EPS:
                            sizebucket_to_sldn_flowsim[idx_1,idx_2*self.model.y_len:(idx_2+1)*self.model.y_len]=self.model.const_tensor
                        
            if self.model.enable_context:
                idx_start = 0
                sizebucket_to_sldn_foreground = sizebucket_to_sldn.new(
                    len(spec), self.model.feat_dim
                )
                sizebucket_to_sldn_context = sizebucket_to_sldn.new(
                    len(spec), self.model.n_embd
                )
                for i in range(len(spec)):
                    sizebucket_to_sldn_foreground[i] = sizebucket_to_sldn_flowsim[
                        idx_start
                    ]
                    idx_interval = sizebucket_to_sldn_flowsim_idx[i]
                    tmp = sizebucket_to_sldn_flowsim[
                        idx_start + 1 : idx_start + idx_interval
                    ]
                    # tmp=torch.flatten(tmp).long()
                    sizebucket_to_sldn_background, _ = self.model.model_transformer(
                        tmp[None, :]
                    )
                    # print("sizebucket_to_sldn_background: ",sizebucket_to_sldn_background.shape)
                    # for j in range(sizebucket_to_sldn_background.shape[1]):
                    #     print(f"logit-{i}-{j}: {tmp[j,0]}, {tmp[j,-2]}")
                    #     print(f"logit-{i}-{j}: {sizebucket_to_sldn_background[0, j,0]}, {sizebucket_to_sldn_background[0, j,-1]}")
                    sizebucket_to_sldn_context[i] = torch.mean(
                        sizebucket_to_sldn_background, dim=1
                    )
                    # print(f"sizebucket_to_sldn_context-{i}: {sizebucket_to_sldn_context[i,0]}, {sizebucket_to_sldn_context[i,-1]}")
                    idx_start += idx_interval

                sizebucket_to_sldn_input = torch.cat(
                    [sizebucket_to_sldn_foreground, sizebucket_to_sldn_context], dim=-1
                )
                #print(f"sizebucket_to_sldn_input: {sizebucket_to_sldn_input[0,0]}, {sizebucket_to_sldn_input[0,300]}, {sizebucket_to_sldn_input[0,301]}, {sizebucket_to_sldn_input[0,-1]}")
            else:
                sizebucket_to_sldn_foreground = sizebucket_to_sldn_flowsim[:, 0, :]
                sizebucket_to_sldn_input = sizebucket_to_sldn_foreground
            # sizebucket_to_sldn_input=torch.cat([sizebucket_to_sldn_input, src_dst_pair_target], dim=-1)
            sizebucket_to_sldn_est = self.model.model_mlp(sizebucket_to_sldn_input)
            sizebucket_to_sldn_est.add_(1.0)
            
            #print(f"sizebucket_to_sldn_est: {sizebucket_to_sldn_est[0,0]},  {sizebucket_to_sldn_est[0,-1]}")
            
            test_dir = f"{self.model.save_dir}/{spec[0]}"
            # logging.info(f"save to {test_dir}")
            os.makedirs(test_dir, exist_ok=True)
            sizebucket_to_sldn_flowsim = sizebucket_to_sldn_flowsim.cpu().numpy()[0]
            sizebucket_to_sldn_input = sizebucket_to_sldn_input.cpu().numpy()[0]
            sizebucket_to_sldn_est = sizebucket_to_sldn_est.cpu().numpy()[0]
            sizebucket_to_sldn = sizebucket_to_sldn.cpu().numpy()
            num_flows_per_cell = num_flows_per_cell.cpu().numpy()
            
            np.savetxt(f'{self.output_dir}/{spec[0]}/feat_output_py.txt', sizebucket_to_sldn_input, fmt='%f',newline=' ')
            # sizebucket_to_sldn_est=sizebucket_to_sldn_est.reshape(x_len_output,y_len)
            # sizebucket_to_sldn=sizebucket_to_sldn.reshape(x_len_output,y_len)
            
            # Concatenate arrays vertically
            # concatenated_array = np.vstack((sizebucket_to_sldn_est, sizebucket_to_sldn))
            # l2_norm = np.linalg.norm(concatenated_array)
            # print(f"{spec[0]}: {l2_norm}")
            # Save the concatenated array to a text file
            # np.savetxt(f'{test_dir}/m3_python.txt', concatenated_array, fmt='%f')
            
def run_m3_inference(config_path, idx):
    """
    调用 m3_inference 类并执行推理。

    参数:
    - config_path: 配置文件的路径
    - idx: 数据列表中要推理的索引

    返回:
    - 推理结果
    """
    # 初始化 m3_inference 类
    inference_instance = m3_inference(config_path)
    
    # 执行推理
    inference_instance.run_inference(idx)

# 示例调用
config_path = "/users/tingzhou/m3/tmp.yaml"
index = 0  # 假设你想推理第一个数据
run_m3_inference(config_path, index)



'''
这个api非常简单，因为参数传递只需要把config_path传入，其他的包括'model_path','inference_data_input','data_output'等
参数都直接在config.yaml文件里配置好就行。
不过我现在不是很清楚到时候用于输入的数据格式run_inference()的参数'idx'在m3中是为了指定一个随机的数据片进行simulation的
具体举例：在生成原始训练数据的时候分片有'shard0_nflows100_nhosts3_lr10Gbps'-'shard99_nflows100_nhosts3_lr10Gbps',而在模拟的时候
就是拿f"shard{idx}_nflows100_nhosts3_lr10Gbps"去跑，我估计之后接起来的时候是用自己的数据跑所以这个idx可能就不必要了..
总之api差不多就这样很简单的。
'''
