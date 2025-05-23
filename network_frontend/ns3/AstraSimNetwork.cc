#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <json/json.hpp>

#include "entry.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "astra-sim/common/Logging.hh"

using namespace std;
using namespace ns3;
using json = nlohmann::json;


/**
 * @class NS3BackendCompletionTracker
 * @brief Tracks the completion status of ranks in the NS3 backend.
 *
 * This is a hacky approach to track which ranks have completed their workload.
 * The purpose of this class is to end the ns3 simulation once all ranks have completed. 
 * The hacky approach is necessary because each ASTRASimNetwork instance only corresponds to one rank.
 * That is, someone needs to keep track of the completion status of all of the ranks.
 * Because there is no exit point once the ns3 simulator has started, 
 * we cannot implement such tracker in the main function.
 */
class NS3BackendCompletionTracker {
    public: 
        NS3BackendCompletionTracker(int num_ranks) {
            num_unfinished_ranks_ = num_ranks;
            completion_tracker_ = vector<int>(num_ranks, 0);
        }

        void mark_rank_as_finished(int rank) {
            if (completion_tracker_[rank] == 0) {
                completion_tracker_[rank] = 1;
                num_unfinished_ranks_--;
            }
            if (num_unfinished_ranks_ == 0) {
                AstraSim::LoggerFactory::get_logger("network")
                    ->debug("All ranks have finished. Exiting simulation.");
                //cout << "All ranks have finished. Exiting simulation.\n";
                Simulator::Stop();
                Simulator::Destroy();
                exit(0);
            }
        }

    private:
        int num_unfinished_ranks_;
        vector<int> completion_tracker_;
};

class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
  public:
    ASTRASimNetwork(int rank, NS3BackendCompletionTracker *completion_tracker) : AstraNetworkAPI(rank) {
        completion_tracker_ = completion_tracker;
    }

    ~ASTRASimNetwork() {}

    void sim_notify_finished() {
        // Output to file instead of stdout
        /*
        for (auto it = node_to_bytes_sent_map.begin();
             it != node_to_bytes_sent_map.end(); it++) {
            pair<int, int> p = it->first;
            if (p.second == 0) {
                cout << "All data sent from node " << p.first << " is "
                     << it->second << "\n";
            } else {
                cout << "All data received by node " << p.first << " is "
                     << it->second << "\n";
            }
        }
        */
        completion_tracker_->mark_rank_as_finished(rank);
        return;
    }

    double sim_time_resolution() {
        return 0;
    }

    void handleEvent(int dst, int cnt) {}

    AstraSim::timespec_t sim_get_time() {
        AstraSim::timespec_t timeSpec;
        timeSpec.time_res = AstraSim::NS;
        timeSpec.time_val = Simulator::Now().GetNanoSeconds();
        return timeSpec;
    }

    virtual void sim_schedule(AstraSim::timespec_t delta,
                              void (*fun_ptr)(void* fun_arg),
                              void* fun_arg) {
        Simulator::Schedule(NanoSeconds(delta.time_val), fun_ptr, fun_arg);
        return;
    }

    virtual int sim_send(void* buffer,
                         uint64_t message_size,
                         int type,
                         int dst_id,
                         int tag,
                         AstraSim::sim_request* request,
                         void (*msg_handler)(void* fun_arg),
                         void* fun_arg) {
        int src_id = rank;

        // Trigger ns3 to schedule RDMA QP event.
        send_flow(src_id, dst_id, message_size, msg_handler, fun_arg, tag);
        return 0;
    }

    virtual int sim_recv(void* buffer,
                         uint64_t message_size,
                         int type,
                         int src_id,
                         int tag,
                         AstraSim::sim_request* request,
                         void (*msg_handler)(void* fun_arg),
                         void* fun_arg) {
        int dst_id = rank;
        MsgEvent recv_event =
            MsgEvent(src_id, dst_id, 1, message_size, fun_arg, msg_handler);
        MsgEventKey recv_event_key =
            make_pair(tag, make_pair(recv_event.src_id, recv_event.dst_id));

        if (received_msg_standby_hash.find(recv_event_key) !=
            received_msg_standby_hash.end()) {
            // 1) ns3 has already received some message before sim_recv is
            // called.
            int received_msg_bytes = received_msg_standby_hash[recv_event_key];
            if (received_msg_bytes == message_size) {
                // 1-1) The received message size is same as what we expect.
                // Exit.
                received_msg_standby_hash.erase(recv_event_key);
                recv_event.callHandler();
            } else if (received_msg_bytes > message_size) {
                // 1-2) The node received more than expected.
                // Do trigger the callback handler for this message, but wait
                // for Sys layer to call sim_recv for more messages.
                received_msg_standby_hash[recv_event_key] =
                    received_msg_bytes - message_size;
                recv_event.callHandler();
            } else {
                // 1-3) The node received less than what we expected.
                // Reduce the number of bytes we are waiting to receive.
                received_msg_standby_hash.erase(recv_event_key);
                recv_event.remaining_msg_bytes -= received_msg_bytes;
                sim_recv_waiting_hash[recv_event_key] = recv_event;
            }
        } else {
            // 2) ns3 has not yet received anything.
            if (sim_recv_waiting_hash.find(recv_event_key) ==
                sim_recv_waiting_hash.end()) {
                // 2-1) We have not been expecting anything.
                sim_recv_waiting_hash[recv_event_key] = recv_event;
            } else {
                // 2-2) We have already been expecting something.
                // Increment the number of bytes we are waiting to receive.
                int expecting_msg_bytes =
                    sim_recv_waiting_hash[recv_event_key].remaining_msg_bytes;
                recv_event.remaining_msg_bytes += expecting_msg_bytes;
                sim_recv_waiting_hash[recv_event_key] = recv_event;
            }
        }
        return 0;
    }

    private:
    NS3BackendCompletionTracker* completion_tracker_;
};

// Command line arguments and default values.
string workload_configuration;
string system_configuration;
string network_configuration;
string memory_configuration;
string comm_group_configuration = "empty";
string logical_topology_configuration;
string logging_configuration = "empty";
int num_queues_per_dim = 1;
double comm_scale = 1;
double injection_scale = 1;
bool rendezvous_protocol = false;
auto logical_dims = vector<int>();
int num_npus = 1;
auto queues_per_dim = vector<int>();

// TODO: Migrate to yaml
void read_logical_topo_config(string network_configuration,
                              vector<int>& logical_dims) {
    ifstream inFile;
    inFile.open(network_configuration);
    if (!inFile) {
        cerr << "Unable to open file: " << network_configuration << endl;
        exit(1);
    }

    // Find the size of each dimension.
    json j;
    inFile >> j;
    if (j.contains("logical-dims")) {
        vector<string> logical_dims_str_vec = j["logical-dims"];
        for (auto logical_dims_str : logical_dims_str_vec) {
            logical_dims.push_back(stoi(logical_dims_str));
        }
    }

    // Find the number of all npus.
    stringstream dimstr;
    for (auto num_npus_per_dim : logical_dims) {
        num_npus *= num_npus_per_dim;
        dimstr << num_npus_per_dim << ",";
    }
    cout << "There are " << num_npus << " npus: " << dimstr.str() << "\n";

    queues_per_dim = vector<int>(logical_dims.size(), num_queues_per_dim);
}

// Read command line arguments.
void parse_args(int argc, char* argv[]) {
    CommandLine cmd;
    cmd.AddValue("workload-configuration", "Workload configuration file.",
                 workload_configuration);
    cmd.AddValue("system-configuration", "System configuration file",
                 system_configuration);
    cmd.AddValue("network-configuration", "Network configuration file",
                 network_configuration);
    cmd.AddValue("remote-memory-configuration", "Memory configuration file",
                 memory_configuration);
    cmd.AddValue("comm-group-configuration",
                 "Communicator group configuration file",
                 comm_group_configuration);
    cmd.AddValue("logical-topology-configuration",
                 "Logical topology configuration file",
                 logical_topology_configuration);
    cmd.AddValue("logging-configuration",
                 "Logging configuration file", 
                 logging_configuration);

    cmd.AddValue("num-queues-per-dim", "Number of queues per each dimension",
                 num_queues_per_dim);
    cmd.AddValue("comm-scale", "Communication scale", comm_scale);
    cmd.AddValue("injection-scale", "Injection scale", injection_scale);
    cmd.AddValue("rendezvous-protocol", "Whether to enable rendezvous protocol",
                 rendezvous_protocol);

    cmd.Parse(argc, argv);
}

int main(int argc, char* argv[]) {
    LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
    LogComponentEnable("PacketSink", LOG_LEVEL_INFO);

    cout << "ASTRA-sim + NS3" << endl;

    // Read network config and find logical dims.
    parse_args(argc, argv);
    AstraSim::LoggerFactory::init(logging_configuration);
    read_logical_topo_config(logical_topology_configuration, logical_dims);

    // Setup network & System layer.
    vector<ASTRASimNetwork*> networks(num_npus, nullptr);
    vector<AstraSim::Sys*> systems(num_npus, nullptr);
    Analytical::AnalyticalRemoteMemory* mem =
        new Analytical::AnalyticalRemoteMemory(memory_configuration);
    NS3BackendCompletionTracker* completion_tracker = new NS3BackendCompletionTracker(num_npus);

    for (int npu_id = 0; npu_id < num_npus; npu_id++) {
        networks[npu_id] = new ASTRASimNetwork(npu_id, completion_tracker);
        systems[npu_id] = new AstraSim::Sys(
            npu_id, workload_configuration, comm_group_configuration,
            system_configuration, mem, networks[npu_id], logical_dims,
            queues_per_dim, injection_scale, comm_scale, rendezvous_protocol);
    }

    // Initialize ns3 simulation.
    if (auto ok = setup_ns3_simulation(network_configuration); ok == -1) {
        std::cerr << "Fail to setup ns3 simulation." << std::endl;
        return -1;
    }

    // Tell workload layer to schedule first events.
    for (int i = 0; i < num_npus; i++) {
        systems[i]->workload->fire();
    }

    // Run the simulation by triggering the ns3 event queue.
    Simulator::Run();
    return 0;
}
