/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#ifdef CCL_ENABLE_MPI
#include "atl/mpi/atl_mpi_ctx.hpp"
#endif
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_barrier.hpp"
#include "coll/algorithms/utils/sycl_copy.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/host_task.hpp"
#include "coll/algorithms/utils/transmit/transmit.hpp"
#include "common/utils/ce_utils.hpp"
#include "proxy/proxy.hpp"

#ifdef CCL_ENABLE_UMF
#include "umf/ipc.hpp"
#endif

// sync_ptrs is used for counting in local kernel_barrier
static ccl_kernel_barrier_data kernel_barrier_data;
static ccl_kernel_barrier_data thread_kernel_barrier_data;

// three tmp buffers - 1: work_buf, 2: tmp_send_buf, 3: tmp_recv_buf
constexpr int tmp_bufs_count = ccl_large_tmp_bufs::buf_count;
// tmp_bufs are used as work buf and to copy input/output
static std::array<void *, tmp_bufs_count> tmp_bufs;

static thread_local std::array<void *, tmp_bufs_count> thread_tmp_bufs;

static thread_local bool is_thread_initial_invocation = true;
size_t tmp_buf_size_per_rank = 0;

std::pair<ccl_sched *, ze_handle_exchange_entry *> do_ipc_exchange(ccl_comm *comm,
                                                                   ccl_stream *stream,
                                                                   std::vector<void *> ptrs,
                                                                   bool to_cache) {
    sycl::queue q = stream->get_native_stream();
    bool host_found = false;
    for (auto ptr : ptrs) {
        sycl::usm::alloc alloc_type = sycl::get_pointer_type(ptr, q.get_context());
        if (alloc_type == sycl::usm::alloc::host) {
            host_found = true;
            break;
        }
    }

    int cache_status = -1;
    // Only disable cache if to_cache is false
    if (!to_cache) {
        cache_status = ccl::global_data::env().enable_ze_cache_get_ipc_handles;
        ccl::global_data::env().enable_ze_cache_get_ipc_handles = 0;
        LOG_DEBUG("disabling ze_cache_get_ipc_handles for do_ipc_exchange");
    }
    // Also disable if host pointers are found but to_cache is true
    else if (host_found) {
        cache_status = ccl::global_data::env().enable_ze_cache_get_ipc_handles;
        ccl::global_data::env().enable_ze_cache_get_ipc_handles = 0;
        LOG_DEBUG("disabling ze_cache_get_ipc_handles for do_ipc_exchange (host pointers found)");
    }

    ccl_comm *node_comm = comm->get_node_comm().get();
    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers;

    for (auto ptr : ptrs) {
        in_buffers.emplace_back(ptr, ccl::ze::ipc_mem_type::memory);
    }

    ccl_coll_param param{};
    param.comm = comm;
    param.stream = stream;
    ccl_coll_attr attr{};
    ccl_sched *sched = ccl_sched::create(param, attr);
    ccl::utils::pt2pt_handle_exchange_info info = {};
    int skip_rank = ccl_comm::invalid_rank;

    ze_handle_exchange_entry *exchange_entry =
        new ze_handle_exchange_entry(sched, node_comm, in_buffers, skip_rank, info);
    // start the entry
    exchange_entry->start();
    while (!exchange_entry->is_completed()) {
        exchange_entry->update(); //    128us
    }

    if (host_found || !to_cache) {
        // restore
        ccl::global_data::env().enable_ze_cache_get_ipc_handles = cache_status;
    }

    return { sched, exchange_entry };
}

const std::unordered_map<int, std::vector<void *>> &do_ipc_exchangeExt(
    ccl_comm *comm,
    std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> &hash_table,
    ccl_stream *stream,
    std::vector<void *> ptrs,
    int exchange_id,
    bool is_pt2pt) {
    std::unordered_map<int, std::vector<void *>> *exchange = nullptr;
    {
        static std::mutex hash_table_mutex;
        std::lock_guard<std::mutex> guard(hash_table_mutex);
        exchange = &hash_table[exchange_id];
        (*exchange)[comm->rank()] = std::move(ptrs);
    }
    if (!is_pt2pt) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }
    return *exchange;
}

static int get_num_queues(sycl::queue q, int ordinal) {
    sycl::device dev = q.get_device();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

    uint32_t queue_group_count = 0;
    ze_result_t result =
        zeDeviceGetCommandQueueGroupProperties(ze_dev, &queue_group_count, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        LOG_WARN("zeDeviceGetCommandQueueGroupProperties failed");
        return 0;
    }

    if (ordinal < queue_group_count) {
        ze_command_queue_group_properties_t *queueProperties =
            (ze_command_queue_group_properties_t *)malloc(
                sizeof(ze_command_queue_group_properties_t) * queue_group_count);
        result =
            zeDeviceGetCommandQueueGroupProperties(ze_dev, &queue_group_count, queueProperties);
        if (result != ZE_RESULT_SUCCESS) {
            LOG_WARN("zeDeviceGetCommandQueueGroupProperties failed");
            return 0;
        }
        int n = queueProperties[ordinal].numQueues;
        free(queueProperties);
        return n;
    }
    else {
        return 0;
    }
}

static sycl::queue create_sycl_queue(sycl::queue &q, int ordinal, int index) {
    // TODO: should we use the parameter q or a new queue?
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    // Create Command Queue
    ze_command_queue_desc_t Qdescriptor = {};
    Qdescriptor.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    Qdescriptor.pNext = NULL;
    Qdescriptor.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Qdescriptor.ordinal = ordinal;
    Qdescriptor.index = index;
    Qdescriptor.flags = 0;
    Qdescriptor.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    //ze_command_queue_handle_t ze_cmd_queue = nullptr;
    //ze_result_t result = zeCommandQueueCreate(ze_ctx, ze_dev, &Qdescriptor, &ze_cmd_queue);

    ze_command_list_handle_t ze_imm_cmd_list = nullptr;
    ze_result_t result =
        zeCommandListCreateImmediate(ze_ctx, ze_dev, &Qdescriptor, &ze_imm_cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        LOG_WARN("zeCommandQueueCreate (", ordinal, ",", index, ") failed, returning same queue");
        return q;
    }

#if 0
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::device> InteropDeviceInput{
        ze_dev
    };
    sycl::device InteropDevice =
        sycl::make_device<sycl::backend::ext_oneapi_level_zero>(InteropDeviceInput);
#endif

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> InteropContextInput{
        ze_ctx, std::vector<sycl::device>(1, dev), sycl::ext::oneapi::level_zero::ownership::keep
    };
    sycl::context InteropContext =
        sycl::make_context<sycl::backend::ext_oneapi_level_zero>(InteropContextInput);

    //sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCQ{
    //  ze_cmd_queue, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep};

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCL{
        ze_imm_cmd_list, dev, sycl::ext::oneapi::level_zero::ownership::keep
    };

    //return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCQ, InteropContext);
    return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCL,
                                                                  InteropContext);
}

// number of link copy engines
static int num_lce;

// main copy engine
static sycl::queue &get_q_me() {
    static sycl::queue q_me;
    return q_me;
}

// link copy engine
static std::vector<sycl::queue> &get_q_le() {
    static std::vector<sycl::queue> q_le;
    return q_le;
}

static void create_copy_engine_queues(sycl::queue q) {
    auto q_me = get_q_me();
    auto q_le = get_q_le();

    num_lce = get_num_queues(q, 2);
    LOG_DEBUG("number of link engines : ", num_lce);

    for (int i = 0; i < num_lce; i++) {
        q_le.push_back(create_sycl_queue(q, 2, i));
    }
    q_me = create_sycl_queue(q, 1, 0);
}

static inline size_t roundup_2MB(size_t s) {
    return (s + 2097152 - 1) / 2097152 * 2097152;
}

static size_t calculate_ll_buf_size(sycl::queue q, ccl_comm *comm = NULL) {
    int proc_count;
    if (comm) {
        // allocate small buffer on each communicator
        proc_count = comm->size();
    }
    else {
        //proc_count = ARC_MAX_NUM;
        proc_count = ccl::global_data::get().get_local_proc_count();
    }
    // from Roger's LL path
    // ringSize is RingTransmit's  static variable
    // the ringSize is  regardless of protocol, and
    // ringSize is per peer rank, and it is rounded up to 2MB
    size_t mem_used_1 = proc_count * RingTransmit<int, Rt64_128_PCIE, true>::ringSize;
    mem_used_1 = roundup_2MB(mem_used_1);
    LOG_DEBUG("Allocate LL ring buffer of ringSize: ",
              RingTransmit<int, Rt64_128_PCIE, true>::ringSize,
              ", mem_used_1: ",
              mem_used_1);

    // from Richard's allreduce path, alltoall uses less
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ssize_t dev_id{ ccl::utils::invalid_device_id };
    if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
        CCL_THROW("unable to get global id for device\n");
    }
    size_t mem_used_2 = proc_count *
                        ccl::global_data::get().ze_data->devices[dev_id].total_threads *
                        sizeof(int) * 4; // sizeof message_t
    //mem_used_2 *= 2; // two slots for gather and scatter
    mem_used_2 = roundup_2MB(mem_used_2);
    LOG_DEBUG("Allocate LL ring buffer of total_threads: ",
              ccl::global_data::get().ze_data->devices[dev_id].total_threads,
              ", mem_used_2: ",
              mem_used_2);

    // Roger's path needs two copies of the buffer
    if (mem_used_1 > mem_used_2) {
        return mem_used_1;
    }
    else {
        return mem_used_2;
    }
}

static bool is_proxy_service_requested() {
    const auto &env = ccl::global_data::env();
    return env.sycl_enable_alltoall_scaleout_proxy || env.sycl_enable_allreduce_scaleout_proxy ||
           env.sycl_enable_reduce_scatter_scaleout_proxy ||
           env.sycl_enable_allgatherv_scaleout_proxy || env.sycl_enable_broadcast_scaleout_proxy;
}

void init_proxy_service(sycl::queue &q) {
    // Nobody asked for the proxy, so do not initialize it.
    if (!is_proxy_service_requested()) {
        return;
    }
    // Initialize proxy service, allocate resources, start proxy thread.
    // A host that cannot back the command ring must not fail runs
    // that never touch the proxy, so init failure is reported and left unstarted.
    // Fallback on non-proxy algorithms in case of any proxy algo is requested.
    try {
        ccl::global_data::get().proxy_service->init(q);
        ccl::global_data::get().proxy_service->start();
        LOG_INFO("CCL_PROXY_SERVICE_STARTED: scale-out proxy service is ready");
    }
    catch (const std::exception &e) {
        LOG_WARN("CCL_PROXY_SERVICE_UNAVAILABLE: ", e.what());
    }
}

void coll_init(ccl_comm *comm, ccl_stream *global_stream) {
    static bool is_initial_invocation = true;
    static bool is_proxy_service_initialization_attempted = false;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    ccl_comm_barrier_data bd = node_comm->barrier_data();

    LOG_DEBUG("coll_init: comm size: ",
              comm->size(),
              " node size: ",
              node_comm->size(),
              " is_initial_invocation :",
              is_initial_invocation);

    sycl::queue q = global_stream->get_native_stream();

    // Proxy setup is process-wide and must not be tied to the communicator's
    // IPC/barrier state. A pre-initialized communicator can have bd.is_set()
    // before this process reaches its first coll_init invocation; keeping proxy
    // setup inside that branch silently leaves a requested proxy unavailable.
    if (!is_proxy_service_initialization_attempted) {
        is_proxy_service_initialization_attempted = true;
        init_proxy_service(q);
    }

    bool is_arc = is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));

    // if communicator is used for first time then do ipc exchage
    // to get remote ptrs used for barrier counting and remote tmp bufs
    if (!bd.is_set()) {
        std::shared_ptr<ccl_comm> numa_comm = comm->get_numa_comm();
        std::shared_ptr<ccl_comm> pcie_switch_comm = comm->get_pcie_switch_comm();
        std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
        std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
        std::vector<std::shared_ptr<ccl_comm>> sub_comms{
            node_comm, numa_comm, pcie_switch_comm, even_comm, pair_comm
        };
        // set atomics capability, this does not seem to work on CI machine
        for (auto sub_comm : sub_comms) {
            sub_comm->set_remote_atomics(node_comm->get_topo_manager().has_p2p_atomics());
        }

        sycl::queue q_worker(q.get_context(), q.get_device());

        // alloc sync pointers to be used for global comm_barrier
        // and p2p_barrier across ranks.
        // each comm_barrier requires maxcimum of node_comm->size()
        // long integer in case remote atomics is not supported.
        // and there is num_slots of this, on top of that each
        // subcomm also have a comm_barrier.
        // Following comm_barrier is the flag data for p2p_barrier.
        // It has only one long integer with a numer of slots, and
        // for each subcomm.
        constexpr int num_slots = ccl_comm_barrier_data::slots;
        // One cell per (channel, writer rank): p2p_barrier_pair_wg() lets several
        // work-groups run the pairwise barrier independently, each on its own channel.
        constexpr size_t pair_flags_per_rank = MAX_NODE_RANKS * MAX_P2P_CHANNELS;
        const size_t ptr_count_comm = node_comm->size() * num_slots;
        const size_t ptr_count_flag = ccl_comm_flag_data::slots * sub_comms.size();
        const size_t ptr_count_pair = pair_flags_per_rank * sub_comms.size();
        size_t ptr_count = ptr_count_comm * sub_comms.size() + ptr_count_flag + ptr_count_pair;
        ptr_count = roundup_2MB(ptr_count * sizeof(size_t)) / sizeof(size_t);
        size_t *ptrs0;
#ifdef CCL_ENABLE_UMF
        if (ccl::global_data::env().umf_enable && node_comm->get_topo_manager().has_p2p_access()) {
            ptrs0 = static_cast<size_t *>(umf_alloc_device(q, ptr_count * sizeof(size_t)));
        }
        else
#endif
            if (node_comm->get_topo_manager().has_p2p_access()) {
            ptrs0 = sycl::aligned_alloc_device<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        else {
            ptrs0 = sycl::aligned_alloc_host<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        q.memset(ptrs0, 0, ptr_count * sizeof(size_t)).wait();

        size_t *ptrs1;
#ifdef CCL_ENABLE_UMF
        if (ccl::global_data::env().umf_enable && node_comm->get_topo_manager().has_p2p_access()) {
            ptrs1 = static_cast<size_t *>(umf_alloc_device(q, ptr_count * sizeof(size_t)));
        }
        else
#endif
            if (node_comm->get_topo_manager().has_p2p_access()) {
            ptrs1 = sycl::aligned_alloc_device<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        else {
            ptrs1 = sycl::aligned_alloc_host<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        q.memset(ptrs1, 0, ptr_count * sizeof(size_t)).wait();

        std::vector<void *> ipc_ptrs;
        for (int i = 0; i < sub_comms.size(); i++) {
            ipc_ptrs.push_back(ptrs0 + i * ptr_count_comm);
        }
        for (int i = 0; i < sub_comms.size(); i++) {
            ipc_ptrs.push_back(ptrs1 + i * ptr_count_comm);
        }
        size_t *ptrs_flag = ptrs1 + ptr_count_comm * sub_comms.size();
        for (int i = 0; i < sub_comms.size(); i++) {
            ipc_ptrs.push_back(ptrs_flag + i * ccl_comm_flag_data::slots);
        }

        // IPC-visible pair_flags cells, one pair_flags_per_rank slice per sub-comm.
        // Located immediately after the flag data region in ptrs1.
        size_t *ptrs_pair = ptrs_flag + ptr_count_flag;
        for (int i = 0; i < sub_comms.size(); i++) {
            ipc_ptrs.push_back(ptrs_pair + i * pair_flags_per_rank);
        }

        // set up cb events to be used for copy engine sync
        const int cb_events_ipc_idx = ipc_ptrs.size();

        bool is_arc = is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));

        if (ccl::global_data::env().enable_ze_ce) {
            // allocate cb events storage for copy engine sync, one per rank in the node.
            // owned by node_comm; it rides the ipc_ptrs exchange below so this condition
            // must stay identical at the retrieval site - an entry's ipc handle index is
            // its position in ipc_ptrs and that layout must match on every rank.
            if (is_arc) {
                LOG_INFO("ARC card detected, init counter based events - rank: ",
                         node_comm->rank(),
                         " node size: ",
                         node_comm->size());
                ipc_ptrs.push_back(node_comm->allocate_cb_events_memory(q));
            }
        }

        // Do one-time initializations that depend on first-time IPC setup.
        if (is_initial_invocation) {
            is_initial_invocation = false;

            create_copy_engine_queues(q);

            // allocate sync_ptrs for local kernel barrier
            size_t *sync_ptrs;
#ifdef CCL_ENABLE_UMF
            if (ccl::global_data::env().umf_enable &&
                node_comm->get_topo_manager().has_p2p_access()) {
                sync_ptrs = static_cast<size_t *>(
                    umf_alloc_device(q, ccl_kernel_barrier_data::slots * sizeof(size_t)));
            }
            else
#endif
                if (node_comm->get_topo_manager().has_p2p_access()) {
                sync_ptrs = sycl::malloc_device<size_t>(ccl_kernel_barrier_data::slots, q);
            }
            else {
                sync_ptrs = sycl::malloc_host<size_t>(ccl_kernel_barrier_data::slots, q);
            }

            // Initialize memory to zero using memset
            q.memset(sync_ptrs, 0, ccl_kernel_barrier_data::slots * sizeof(size_t)).wait();

            // Set the sync_ptrs for the kernel barrier data
            get_kernel_barrier_data(comm).set_sync_ptrs(sync_ptrs);

            //set up temp buf to be used for large collectives
            // WA : use smaller tmp buffer for client GPUs
            if (is_arc && ccl::global_data::env().sycl_ll_buffer_global &&
                ccl::global_data::env().sycl_tmp_buf_size == tmp_bufs_count * 128 * 1024 * 1024) {
                ccl::global_data::env().sycl_tmp_buf_size =
                    tmp_bufs_count * calculate_ll_buf_size(q, NULL);
                LOG_DEBUG("Allocate LL ring buffer of size: ",
                          ccl::global_data::env().sycl_tmp_buf_size);
            }
            const size_t tmp_buf_size =
                ccl::global_data::env().sycl_tmp_buf_size / tmp_bufs_count / large_tmp_buf_align *
                large_tmp_buf_align;
            // make sure sycl_tmp_buf_size can be divided by buf_count
            ccl::global_data::env().sycl_tmp_buf_size = tmp_buf_size * tmp_bufs_count;
            const size_t tmp_buf_size_per_rank_orig =
                tmp_buf_size / ccl::global_data::get().get_local_proc_count();

            // adjust tmp_buf_size_per_rank to align in all ranks
            const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
            tmp_buf_size_per_rank = (tmp_buf_size_per_rank_orig / align_bytes) * align_bytes;
            LOG_DEBUG("Adjusted large temp buf size: ",
                      ccl::global_data::env().sycl_tmp_buf_size,
                      ", tmp_buf_size_per_rank: ",
                      tmp_buf_size_per_rank);

            char *tmp_buf;
#ifdef CCL_ENABLE_UMF
            if (ccl::global_data::env().umf_enable &&
                node_comm->get_topo_manager().has_p2p_access()) {
                tmp_buf = static_cast<char *>(umf_alloc_device_aligned(
                    q, CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count));
            }
            else
#endif
                if (node_comm->get_topo_manager().has_p2p_access()) {
                tmp_buf = sycl::aligned_alloc_device<char>(
                    CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);
            }
            else {
                tmp_buf = sycl::aligned_alloc_host<char>(
                    CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);
            }

            for (int i = 0; i < tmp_bufs_count; i++) {
                tmp_bufs[i] = tmp_buf + i * tmp_buf_size;
            }
        }

        for (auto sub_comm : sub_comms) {
            ccl_large_tmp_bufs &comm_large_tmp_bufs = sub_comm->get_large_tmp_bufs();
            for (int i = 0; i < tmp_bufs_count; i++) {
                comm_large_tmp_bufs.tmp_bufs[i] = tmp_bufs[i];
            }
        }

        // set up temp buf to be used for small collectives
        const int small_buf_ipc_idx = ipc_ptrs.size();
        if (is_arc && !ccl::global_data::env().sycl_ll_buffer_global) {
            size_t tmp_buf_size = node_comm->get_tmp_buf_size();
            size_t arc_size = calculate_ll_buf_size(q, node_comm.get()) * 2;
            if (arc_size > tmp_buf_size) {
                tmp_buf_size = arc_size;
                for (auto sub_comm : sub_comms) {
                    sub_comm->set_tmp_buf_size(tmp_buf_size);
                }
            }
            LOG_DEBUG("Allocate small buffer for LL ring of size: ",
                      tmp_buf_size,
                      " x ",
                      ccl_tmp_bufs::buf_count);
        }

        // only node_comm allocate buffers
        node_comm->allocate_buffers(
            q, node_comm->get_topo_manager().has_p2p_access(), ccl::global_data::env().umf_enable);

        // propagate the tmp buffer pointers to all subcomms
        for (auto sub_comm : sub_comms) {
            if (sub_comm == node_comm)
                continue;
            sub_comm->inherit_pointers(node_comm.get());
        }

        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            ipc_ptrs.push_back(node_comm->get_tmp_buf(i));
        }

        const int small_buf_ipc_gpu_idx = ipc_ptrs.size();
        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            ipc_ptrs.push_back(node_comm->get_tmp_buf_gpu(i));
        }

        // add tmp buf pointers of large buffers
        const int large_buf_ipc_idx = ipc_ptrs.size();
        ipc_ptrs.push_back(tmp_bufs[0]);

#ifdef CCL_ENABLE_UMF
        if (ccl::global_data::env().umf_enable) {
            LOG_DEBUG("|UMF|: umf_ipc_exchange");
            umf_ipc_exchange(comm, global_stream, ipc_ptrs);

            // add comm_barrier sync pointers to each communicator
            size_t sub_comms_size = sub_comms.size();
            size_t *counter = sycl::malloc_device<size_t>(sub_comms_size * 2, q);
            q.fill((void *)counter, static_cast<size_t>(num_slots - 1), sub_comms_size).wait();
            size_t *counter_flag = counter + sub_comms_size;
            q.fill(counter_flag, static_cast<size_t>(ccl_comm_flag_data::slots - 1), sub_comms_size)
                .wait();
            size_t *counter_pair = counter_flag + sub_comms_size;
            q.fill(counter_pair, static_cast<size_t>(0), MAX_NODE_RANKS * sub_comms_size).wait();
            for (size_t i = 0, j = sub_comms_size; i < sub_comms.size(); i++, j++) {
                size_t ptrs0_idx = i;
                size_t ptrs1_idx = j;
                size_t ptrs2_idx = j + sub_comms_size;
                auto remote_ptrs0 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs0_idx,
                                                                         ipc_ptrs[ptrs0_idx],
                                                                         ipc_handle_map,
                                                                         q_worker,
                                                                         q.get_device(),
                                                                         1);
                auto remote_ptrs1 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs1_idx,
                                                                         ipc_ptrs[ptrs1_idx],
                                                                         ipc_handle_map,
                                                                         q_worker,
                                                                         q.get_device(),
                                                                         1);
                auto remote_ptrs2 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs2_idx,
                                                                         ipc_ptrs[ptrs2_idx],
                                                                         ipc_handle_map,
                                                                         q_worker,
                                                                         q.get_device(),
                                                                         1);
                const int ptrs_pair_idx = sub_comms.size() * 3 + i;
                auto remote_pair_ptrs =
                    get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                         ptrs_pair_idx,
                                                         ipc_ptrs[ptrs_pair_idx],
                                                         ipc_handle_map,
                                                         q_worker,
                                                         q.get_device(),
                                                         1);
                sub_comms[i]->set_barrier_ptrs(remote_ptrs0, remote_ptrs1, counter + i);
                sub_comms[i]->set_flag_ptrs(remote_ptrs2, counter_flag + i);
                sub_comms[i]->set_pair_flag_ptrs(remote_pair_ptrs,
                                                 counter_pair + i * MAX_NODE_RANKS);
            }
            if (ccl::global_data::env().enable_ze_ce) {
                // add cb events pointers to node_comm for copy engine sync
                if (is_arc) {
                    LOG_INFO(
                        "UMF enabled ARC card detected, setting remote ranks counter based events - rank: ",
                        node_comm->rank(),
                        " node size: ",
                        node_comm->size());
                    auto device_ptrs =
                        get_ipc_ptrs<void, MAX_NODE_RANKS>(node_comm,
                                                           cb_events_ipc_idx,
                                                           ipc_ptrs[cb_events_ipc_idx],
                                                           ipc_handle_map,
                                                           q_worker,
                                                           q.get_device(),
                                                           1);
                    node_comm->init_cb_events(device_ptrs,
                                              ce_utils_t::get_ze_device(q, comm->get_devices()));
                }
            }
            // get ipc pointers for small tmp buffers and add them to node_comm
            for (auto sub_comm : sub_comms) {
                for (size_t i = 0, j = small_buf_ipc_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
                    auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                        sub_comm, j, ipc_ptrs[j], ipc_handle_map, q_worker, q.get_device(), 1);
                    sub_comm->set_remote_tmp_bufs(remote_ptrs, i);
                }
                for (size_t i = 0, j = small_buf_ipc_gpu_idx; i < ccl_tmp_bufs::buf_count;
                     i++, j++) {
                    auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                        sub_comm, j, ipc_ptrs[j], ipc_handle_map, q_worker, q.get_device(), 1);
                    sub_comm->set_remote_tmp_bufs_gpu(remote_ptrs, i);
                }
            }

            // node_comm
            const size_t tmp_buf_size = ccl::global_data::env().sycl_tmp_buf_size / tmp_bufs_count;
            for (auto sub_comm : sub_comms) {
                ccl_large_tmp_bufs &comm_large_tmp_bufs = sub_comm->get_large_tmp_bufs();
                comm_large_tmp_bufs.remote_tmp_bufs[0] =
                    get_ipc_ptrs<void, MAX_NODE_RANKS>(sub_comm,
                                                       large_buf_ipc_idx,
                                                       ipc_ptrs[large_buf_ipc_idx],
                                                       ipc_handle_map,
                                                       q_worker,
                                                       q.get_device(),
                                                       1);
                for (int i = 0; i < sub_comm->size(); i++) {
                    for (int j = 1; j < tmp_bufs_count; j++) {
                        comm_large_tmp_bufs.remote_tmp_bufs[j][i] =
                            (char *)(comm_large_tmp_bufs.remote_tmp_bufs[0][i]) + tmp_buf_size * j;
                    }
                }
            }

            q_worker.wait();
        }
        else {
#endif // CCL_ENABLE_UMF

            LOG_DEBUG("|SCHED|: do_ipc_exchange: with sched");
            auto [sched, exchange_entry] =
                do_ipc_exchange(comm, global_stream, ipc_ptrs, false /* to_cache */);

            // add comm_barrier sync pointers to each communicator
            size_t sub_comms_size = sub_comms.size();
            // Three regions: one counter per sub-comm for comm_barrier, one per sub-comm for the
            // flag data, then pair_flags_per_rank counters per sub-comm for the pairwise barrier.
            // That last region was filled but left out of the allocation size, overrunning it.
            size_t *counter =
                sycl::malloc_device<size_t>(sub_comms_size * (2 + pair_flags_per_rank), q);
            q.fill((void *)counter, static_cast<size_t>(num_slots - 1), sub_comms_size).wait();
            size_t *counter_flag = counter + sub_comms_size;
            q.fill(counter_flag, static_cast<size_t>(ccl_comm_flag_data::slots - 1), sub_comms_size)
                .wait();
            size_t *counter_pair = counter_flag + sub_comms_size;
            q.fill(counter_pair, static_cast<size_t>(0), pair_flags_per_rank * sub_comms_size)
                .wait();
            for (size_t i = 0, j = sub_comms_size; i < sub_comms.size(); i++, j++) {
                size_t ptrs0_idx = i;
                size_t ptrs1_idx = j;
                size_t ptrs2_idx = j + sub_comms_size;
                auto remote_ptrs0 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs0_idx,
                                                                         ipc_ptrs[ptrs0_idx],
                                                                         sched,
                                                                         q_worker,
                                                                         1,
                                                                         false /* to_cache */);
                auto remote_ptrs1 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs1_idx,
                                                                         ipc_ptrs[ptrs1_idx],
                                                                         sched,
                                                                         q_worker,
                                                                         1,
                                                                         false /* to_cache */);
                auto remote_ptrs2 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs2_idx,
                                                                         ipc_ptrs[ptrs2_idx],
                                                                         sched,
                                                                         q_worker,
                                                                         1,
                                                                         false /* to_cache */);
                const int ptrs_pair_idx = sub_comms.size() * 3 + i;
                auto remote_pair_ptrs =
                    get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                         ptrs_pair_idx,
                                                         ipc_ptrs[ptrs_pair_idx],
                                                         sched,
                                                         q_worker,
                                                         1,
                                                         false /* to_cache */);
                sub_comms[i]->set_barrier_ptrs(remote_ptrs0, remote_ptrs1, counter + i);
                sub_comms[i]->set_flag_ptrs(remote_ptrs2, counter_flag + i);
                sub_comms[i]->set_pair_flag_ptrs(remote_pair_ptrs,
                                                 counter_pair + i * pair_flags_per_rank);
            }

            if (ccl::global_data::env().enable_ze_ce) {
                // add cb events pointers to node_comm for copy engine sync
                if (is_arc) {
                    LOG_INFO(
                        "UMF disabled ARC card detected, setting remote ranks counter based events - rank: ",
                        node_comm->rank(),
                        " node size: ",
                        node_comm->size());
                    auto device_ptrs =
                        get_ipc_ptrs<void, MAX_NODE_RANKS>(node_comm,
                                                           cb_events_ipc_idx,
                                                           ipc_ptrs[cb_events_ipc_idx],
                                                           sched,
                                                           q_worker,
                                                           1,
                                                           false);
                    node_comm->init_cb_events(device_ptrs,
                                              ce_utils_t::get_ze_device(q, comm->get_devices()));
                }
            }

            // get ipc pointers for small tmp buffers and add them to node_comm
            for (auto sub_comm : sub_comms) {
                for (size_t i = 0, j = small_buf_ipc_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
                    auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                        sub_comm, j, ipc_ptrs[j], sched, q_worker, 1, false /* to_cache */);
                    sub_comm->set_remote_tmp_bufs(remote_ptrs, i);
                }
                // get ipc pointers for small tmp gpu buffers and add them to node_comm
                for (size_t i = 0, j = small_buf_ipc_gpu_idx; i < ccl_tmp_bufs::buf_count;
                     i++, j++) {
                    auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                        sub_comm, j, ipc_ptrs[j], sched, q_worker, 1, false /* to_cache */);
                    sub_comm->set_remote_tmp_bufs_gpu(remote_ptrs, i);
                }
            }
            // get ipc pointers for large tmp_buffers
            const size_t tmp_buf_size = ccl::global_data::env().sycl_tmp_buf_size / tmp_bufs_count;
            // node_comm
            for (auto sub_comm : sub_comms) {
                ccl_large_tmp_bufs &comm_large_tmp_bufs = sub_comm->get_large_tmp_bufs();
                comm_large_tmp_bufs.remote_tmp_bufs[0] =
                    get_ipc_ptrs<void, MAX_NODE_RANKS>(sub_comm,
                                                       large_buf_ipc_idx,
                                                       ipc_ptrs[large_buf_ipc_idx],
                                                       sched,
                                                       q_worker,
                                                       1,
                                                       false /* to_cache */);
                for (int i = 0; i < sub_comm->size(); i++) {
                    for (int j = 1; j < tmp_bufs_count; j++) {
                        comm_large_tmp_bufs.remote_tmp_bufs[j][i] =
                            (char *)(comm_large_tmp_bufs.remote_tmp_bufs[0][i]) + tmp_buf_size * j;
                    }
                }
            }

            q_worker.wait();

            comm->set_handle_exchange_data(std::make_shared<HandleExchangeData>(
                std::shared_ptr<ze_handle_exchange_entry>(exchange_entry),
                std::shared_ptr<ccl_sched>(sched)));

#ifdef CCL_ENABLE_UMF
        }
#endif // CCL_ENABLE_UMF

        auto evt = ccl_host_task(q, {}, [node_comm]() {
            comm_barrier(node_comm);
        });
        evt.wait();
    }
}

void coll_initExt(ccl_comm *comm,
                  std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> &hash_table,
                  ccl_stream *global_stream) {
    // TODO: check all pthread_barrier_wait invokes
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    CCL_THROW_IF_NOT(!use_recording_path(global_stream),
                     "|CCL_SYCL| Multi-threaded algos do not support sycl_graph recording");
    ccl_comm_barrier_data bd = node_comm->barrier_data();
    LOG_DEBUG("coll_initExt: comm rank=",
              comm->rank(),
              " size=",
              comm->size(),
              " global_id=",
              comm->global_current_id,
              " node_comm size=",
              node_comm->size(),
              " rank=",
              node_comm->rank(),
              " bd.is_set=",
              bd.is_set());
    pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));

    bool is_arc = is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));

    // if communicator is used for the first time then do ipc exchange
    // to get remote pointers used for barrier counting and remote tmp bufs
    if (!bd.is_set()) {
        // numa comm and pcie switch comm are not supported
        //std::shared_ptr<ccl_comm> numa_comm = comm->get_numa_comm();
        std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
        std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
        std::vector<std::shared_ptr<ccl_comm>> sub_comms{ node_comm, even_comm, pair_comm };
        // set atomics capability, this does not seem to work on CI machine
        for (auto comm : sub_comms) {
            comm->set_remote_atomics(node_comm->get_topo_manager().has_p2p_atomics());
        }

        sycl::queue q = global_stream->get_native_stream();
        sycl::queue q_worker(q.get_context(), q.get_device());

        // alloc sync pointers to be used for global comm_barrier across ranks
        constexpr int num_slots = ccl_comm_barrier_data::slots;
        constexpr size_t pair_flags_per_rank = MAX_NODE_RANKS;
        const size_t ptr_count_comm = node_comm->size() * num_slots;
        const size_t ptr_count_flag = ccl_comm_flag_data::slots * sub_comms.size();
        const size_t ptr_count_pair = pair_flags_per_rank * sub_comms.size();
        size_t ptr_count = ptr_count_comm * sub_comms.size() + ptr_count_flag + ptr_count_pair;
        ptr_count = roundup_2MB(ptr_count * sizeof(size_t)) / sizeof(size_t);
        size_t *ptrs0;
#ifdef CCL_ENABLE_UMF
        if (ccl::global_data::env().umf_enable && node_comm->get_topo_manager().has_p2p_access()) {
            ptrs0 = static_cast<size_t *>(umf_alloc_device(q, ptr_count * sizeof(size_t)));
        }
        else
#endif
            if (node_comm->get_topo_manager().has_p2p_access()) {
            ptrs0 = sycl::aligned_alloc_device<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        else {
            ptrs0 = sycl::aligned_alloc_host<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        q.memset(ptrs0, 0, ptr_count * sizeof(size_t)).wait();

        size_t *ptrs1;
#ifdef CCL_ENABLE_UMF
        if (ccl::global_data::env().umf_enable && node_comm->get_topo_manager().has_p2p_access()) {
            ptrs1 = static_cast<size_t *>(umf_alloc_device(q, ptr_count * sizeof(size_t)));
        }
        else
#endif
            if (node_comm->get_topo_manager().has_p2p_access()) {
            ptrs1 = sycl::aligned_alloc_device<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        else {
            ptrs1 = sycl::aligned_alloc_host<size_t>(CCL_REG_MSG_ALIGNMENT, ptr_count, q);
        }
        q.memset(ptrs1, 0, ptr_count * sizeof(size_t)).wait();

        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));

        std::vector<void *> ipc_ptrs;
        for (int i = 0; i < sub_comms.size(); i++) {
            ipc_ptrs.push_back(ptrs0 + i * ptr_count_comm);
        }
        for (int i = 0; i < sub_comms.size(); i++) {
            ipc_ptrs.push_back(ptrs1 + i * ptr_count_comm);
        }
        size_t *ptrs_flag = ptrs1 + ptr_count_comm * sub_comms.size();
        for (int i = 0; i < sub_comms.size(); i++) {
            ipc_ptrs.push_back(ptrs_flag + i * ccl_comm_flag_data::slots);
        }

        // do one-time initializations
        if (is_thread_initial_invocation) {
            is_thread_initial_invocation = false;

            // allocate sync_ptrs for local kernel barrier
            size_t *sync_ptrs;
#ifdef CCL_ENABLE_UMF
            if (ccl::global_data::env().umf_enable &&
                node_comm->get_topo_manager().has_p2p_access()) {
                sync_ptrs = static_cast<size_t *>(
                    umf_alloc_device(q, ccl_kernel_barrier_data::slots * sizeof(size_t)));
            }
            else
#endif
                if (node_comm->get_topo_manager().has_p2p_access()) {
                sync_ptrs = sycl::malloc_device<size_t>(ccl_kernel_barrier_data::slots, q);
            }
            else {
                sync_ptrs = sycl::malloc_host<size_t>(ccl_kernel_barrier_data::slots, q);
            }

            // Initialize memory to zero using memset
            q.memset(sync_ptrs, 0, ccl_kernel_barrier_data::slots * sizeof(size_t)).wait();

            // Set the sync_ptrs for the kernel barrier data
            get_kernel_barrier_data(comm).set_sync_ptrs(sync_ptrs);

            //set up temp buf to be used for large collectives
            // WA : use smaller tmp buffer for client GPUs
            if (is_arc &&
                ccl::global_data::env().sycl_tmp_buf_size == tmp_bufs_count * 128 * 1024 * 1024) {
                ccl::global_data::env().sycl_tmp_buf_size =
                    tmp_bufs_count * calculate_ll_buf_size(q, NULL);
                LOG_DEBUG("MT: Allocate LL ring buffer of size: ",
                          ccl::global_data::env().sycl_tmp_buf_size);
            }
            const size_t tmp_buf_size =
                ccl::global_data::env().sycl_tmp_buf_size / tmp_bufs_count / large_tmp_buf_align *
                large_tmp_buf_align;
            // make sure sycl_tmp_buf_size can be divided by buf_count
            ccl::global_data::env().sycl_tmp_buf_size = tmp_buf_size * tmp_bufs_count;
            const size_t tmp_buf_size_per_rank_orig =
                tmp_buf_size / comm->size(); //ccl::global_data::get().get_local_proc_count();
            // adjust tmp_buf_size_per_rank to align in all ranks
            const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
            tmp_buf_size_per_rank = (tmp_buf_size_per_rank_orig / align_bytes) * align_bytes;

            char *tmp_buf;
#ifdef CCL_ENABLE_UMF
            if (ccl::global_data::env().umf_enable &&
                node_comm->get_topo_manager().has_p2p_access()) {
                tmp_buf = static_cast<char *>(umf_alloc_device_aligned(
                    q, CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count));
            }
            else
#endif
                if (node_comm->get_topo_manager().has_p2p_access()) {
                tmp_buf = sycl::aligned_alloc_device<char>(
                    CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);
            }
            else {
                tmp_buf = sycl::aligned_alloc_host<char>(
                    CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);
            }

            for (int i = 0; i < tmp_bufs_count; i++) {
                thread_tmp_bufs[i] = tmp_buf + i * tmp_buf_size;
            }
        }
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));

        for (auto sub_comm : sub_comms) {
            ccl_large_tmp_bufs &comm_large_tmp_bufs = sub_comm->get_large_tmp_bufs();
            for (int i = 0; i < tmp_bufs_count; i++) {
                comm_large_tmp_bufs.tmp_bufs[i] = thread_tmp_bufs[i];
            }
        }

        // set up temp buf to be used for small collectives
        const int small_buf_ipc_idx = ipc_ptrs.size();
        if (is_arc && !ccl::global_data::env().sycl_ll_buffer_global) {
            size_t tmp_buf_size = node_comm->get_tmp_buf_size();
            size_t arc_size = calculate_ll_buf_size(q, node_comm.get()) * 2;
            if (arc_size > tmp_buf_size) {
                tmp_buf_size = arc_size;
                for (auto comm : sub_comms) {
                    comm->set_tmp_buf_size(tmp_buf_size);
                }
            }
            LOG_DEBUG("MT Allocate small buffer for LL ring of size: ",
                      tmp_buf_size,
                      " x ",
                      ccl_tmp_bufs::buf_count);
        }

        node_comm->allocate_buffers(
            q, node_comm->get_topo_manager().has_p2p_access(), ccl::global_data::env().umf_enable);

        // propagate the tmp buffer pointers to all subcomms
        for (auto sub_comm : sub_comms) {
            if (sub_comm == node_comm)
                continue;
            sub_comm->inherit_pointers(node_comm.get());
        }

        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            ipc_ptrs.push_back(node_comm->get_tmp_buf(i));
        }

        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));

        const int small_buf_ipc_gpu_idx = ipc_ptrs.size();
        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            ipc_ptrs.push_back(node_comm->get_tmp_buf_gpu(i));
        }

        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        // add tmp buf pointers of large buffers
        ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();
        const int large_buf_ipc_idx = ipc_ptrs.size();
        ipc_ptrs.insert(std::end(ipc_ptrs),
                        std::begin(comm_large_tmp_bufs.tmp_bufs),
                        std::end(comm_large_tmp_bufs.tmp_bufs));

        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));

        const auto &exchange =
            do_ipc_exchangeExt(comm, hash_table, global_stream, ipc_ptrs, comm->global_current_id);

        // add comm_barrier sync pointers to each communicator
        size_t sub_comms_size = sub_comms.size();
        size_t *counter = sycl::malloc_device<size_t>(sub_comms_size * 2, q);
        q.fill((void *)counter, static_cast<size_t>(num_slots - 1), sub_comms_size).wait();
        size_t *counter_flag = counter + sub_comms_size;
        q.fill(counter_flag, static_cast<size_t>(ccl_comm_flag_data::slots - 1), sub_comms_size)
            .wait();
        size_t *counter_pair = counter_flag + sub_comms_size;
        q.fill(counter_pair, static_cast<size_t>(0), MAX_NODE_RANKS * sub_comms_size).wait();
        // TODO: sycl graph are not supported for MT
        q.submit([=](sycl::handler &h) {
             // h.single_task([=]() {
             //     // for (int i = 0; i < sub_comms_size; i++) {
             //     //     counter[i] = num_slots - 1;
             //     // }
             // });
         }).wait();

        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        for (size_t i = 0, j = sub_comms_size; i < sub_comms.size(); i++, j++) {
            size_t ptrs0_idx = i;
            size_t ptrs1_idx = j;
            size_t ptrs2_idx = j + sub_comms_size;
            auto remote_ptrs0 =
                get_ipc_ptrsExt<size_t, MAX_NODE_RANKS>(*ccl::global_data::get().shared_data.get(),
                                                        sub_comms[i],
                                                        exchange,
                                                        i,
                                                        ptrs0_idx,
                                                        ipc_ptrs[ptrs0_idx],
                                                        even_comm,
                                                        pair_comm);
            auto remote_ptrs1 =
                get_ipc_ptrsExt<size_t, MAX_NODE_RANKS>(*ccl::global_data::get().shared_data.get(),
                                                        sub_comms[i],
                                                        exchange,
                                                        i,
                                                        ptrs1_idx,
                                                        ipc_ptrs[ptrs1_idx],
                                                        even_comm,
                                                        pair_comm);
            auto remote_ptrs2 =
                get_ipc_ptrsExt<size_t, MAX_NODE_RANKS>(*ccl::global_data::get().shared_data.get(),
                                                        sub_comms[i],
                                                        exchange,
                                                        i,
                                                        ptrs2_idx,
                                                        ipc_ptrs[ptrs2_idx],
                                                        even_comm,
                                                        pair_comm);
            const int ptrs_pair_idx = sub_comms.size() * 3 + i;
            auto remote_pair_ptrs =
                get_ipc_ptrsExt<size_t, MAX_NODE_RANKS>(*ccl::global_data::get().shared_data.get(),
                                                        sub_comms[i],
                                                        exchange,
                                                        0 /*node*/,
                                                        ptrs_pair_idx,
                                                        ipc_ptrs[ptrs_pair_idx],
                                                        even_comm,
                                                        pair_comm);
            // TODO: sycl graph are not supported for MT
            // sub_comms[i]->set_barrier_ptrs(remote_ptrs0, remote_ptrs1, counter + i);
            sub_comms[i]->set_barrier_ptrs(remote_ptrs0, remote_ptrs1, nullptr);
            sub_comms[i]->set_flag_ptrs(remote_ptrs2, counter_flag + i);
            sub_comms[i]->set_pair_flag_ptrs(remote_pair_ptrs, counter_pair + i * MAX_NODE_RANKS);
        }
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        // get ipc pointers for small tmp buffers and add them to node_comm
        for (size_t i = 0, j = small_buf_ipc_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
            auto remote_ptrs =
                get_ipc_ptrsExt<void, MAX_NODE_RANKS>(*ccl::global_data::get().shared_data.get(),
                                                      node_comm,
                                                      exchange,
                                                      0 /*node*/,
                                                      j,
                                                      ipc_ptrs[j],
                                                      even_comm,
                                                      pair_comm);
            node_comm->set_remote_tmp_bufs(remote_ptrs, i);
            LOG_DEBUG("coll_initExt: small buf exchange: rank=",
                      comm->rank(),
                      " global_id=",
                      comm->global_current_id,
                      " buf_idx=",
                      i,
                      " local=",
                      ipc_ptrs[j],
                      " remote[0]=",
                      (void *)remote_ptrs[0],
                      " remote[1]=",
                      (void *)remote_ptrs[1]);
        }

        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        // get ipc pointers for small tmp gpu buffers and add them to node_comm
        for (size_t i = 0, j = small_buf_ipc_gpu_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
            auto remote_ptrs =
                get_ipc_ptrsExt<void, MAX_NODE_RANKS>(*ccl::global_data::get().shared_data.get(),
                                                      node_comm,
                                                      exchange,
                                                      0 /*node*/,
                                                      j,
                                                      ipc_ptrs[j],
                                                      even_comm,
                                                      pair_comm);
            node_comm->set_remote_tmp_bufs_gpu(remote_ptrs, i); // TODO this crashes
        }

        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        // get ipc pointers for large tmp_buffers
        for (size_t i = 0, j = large_buf_ipc_idx; i < tmp_bufs.size(); i++, j++) {
            for (auto sub_comm : sub_comms) {
                ccl_large_tmp_bufs &comm_large_tmp_bufs = sub_comm->get_large_tmp_bufs();
                comm_large_tmp_bufs.remote_tmp_bufs[i] = get_ipc_ptrsExt<void, MAX_NODE_RANKS>(
                    *ccl::global_data::get().shared_data.get(),
                    sub_comm,
                    exchange,
                    0 /*node*/,
                    j,
                    ipc_ptrs[j],
                    even_comm,
                    pair_comm);
            }
        }
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        q_worker.wait();
    }
    pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
}

ccl_kernel_barrier_data &get_kernel_barrier_data(ccl_comm *comm) {
    return comm->is_multi_thread_instance() ? thread_kernel_barrier_data : kernel_barrier_data;
}

void *get_large_tmp_buf(int index, ccl_comm *comm) {
    ccl_comm *node_comm = comm->get_node_comm().get();
    ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();
    return comm_large_tmp_bufs.tmp_bufs[index];
}

std::array<void *, MAX_NODE_RANKS> get_remote_large_tmp_buf(int index, ccl_comm *comm) {
    ccl_large_tmp_bufs &comm_large_tmp_bufs = comm->get_large_tmp_bufs();
    return comm_large_tmp_bufs.remote_tmp_bufs[index];
}

size_t get_tmp_buf_size_per_rank() {
    return tmp_buf_size_per_rank;
}

size_t get_simple_ring_segment_size(const char* env_name,
                                    size_t requested,
                                    int pipeline_depth,
                                    size_t align_bytes) {
    CCL_THROW_IF_NOT(pipeline_depth > 0, "invalid pipeline depth: ", pipeline_depth);
    // sycl_tmp_buf_size holds the rounded allocation total after coll_init
    const size_t max_segment = ccl::global_data::env().sycl_tmp_buf_size / pipeline_depth;
    size_t segment = requested ? requested : max_segment;
    if (segment > max_segment) {
        LOG_WARN_ONCE(env_name,
                      " = ",
                      requested,
                      " exceeds CCL_SYCL_TMP_BUF_SIZE / pipeline depth (",
                      ccl::global_data::env().sycl_tmp_buf_size,
                      " / ",
                      pipeline_depth,
                      "), clamping to ",
                      max_segment,
                      "; ");
        segment = max_segment;
    }
    if (align_bytes > 1) {
        segment = segment / align_bytes * align_bytes;
    }
    CCL_THROW_IF_NOT(segment > 0,
                     env_name,
                     " = ",
                     requested,
                     " is too small for pipeline depth ",
                     pipeline_depth,
                     " and alignment ",
                     align_bytes);
    return segment;
}

std::vector<sycl::event> get_sycl_events(const ccl::vector_class<ccl::event> &deps) {
    std::vector<sycl::event> ret;
    if (!group_impl::is_group_active()) {
        for (auto &dep : deps) {
            ret.push_back(dep.get_native());
        }
    }
    return ret;
}

// invoke the point to point communication barrier kernel
sycl::event invoke_p2p_barrier(const std::shared_ptr<ccl_comm> comm,
                               sycl::queue q,
                               const std::vector<sycl::event> &dep_events,
                               bool use_cpu) {
    sycl::event e;
    if (use_cpu) {
        // host code path uses mpi_comm_barrier rather than point to point
        e = ccl_host_task(q, dep_events, [comm]() {
            comm_barrier(comm);
        });
    }
    else {
        ccl_comm_flag_data flag_data = comm->flag_data();
        e = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.parallel_for(sycl::nd_range<1>(MAX_NODE_RANKS, MAX_NODE_RANKS),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                               p2p_barrier(flag_data, it);
                           });
        });
    }
    return e;
}

sycl::event invoke_p2p_barrier_pair(const std::shared_ptr<ccl_comm> comm,
                                    int peer_rank,
                                    sycl::queue q,
                                    const std::vector<sycl::event> &dep_events) {
    ccl_comm_pair_flag_data pflag = comm->pair_flag_data();
    // Validate peer_rank before submission: the device kernel indexes
    // pair_counts[peer_rank] and d_remote_ptrs[peer_rank] directly, so out-of-
    // bounds values would corrupt memory or dereference uninitialized IPC ptrs.
    CCL_THROW_IF_NOT(peer_rank >= 0 && peer_rank < pflag.size(),
                     "invoke_p2p_barrier_pair: peer_rank ",
                     peer_rank,
                     " out of range [0, ",
                     pflag.size(),
                     ")");
    CCL_THROW_IF_NOT(peer_rank != pflag.rank(),
                     "invoke_p2p_barrier_pair: peer_rank must differ from self rank ",
                     pflag.rank());
    return q.submit([=](sycl::handler &h) {
        h.depends_on(dep_events);
        h.parallel_for(sycl::nd_range<1>(MAX_NODE_RANKS, MAX_NODE_RANKS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                           p2p_barrier_pair(pflag, peer_rank, it);
                       });
    });
}

int get_num_lce() {
    return num_lce;
}

// get main copy engine
sycl::queue get_mce_queue(sycl::queue q) {
    auto q_me = get_q_me();

    return q_me;
}

// get link copy engine
sycl::queue get_lce_queue(sycl::queue q, int index) {
    auto q_le = get_q_le();

    return q_le[index];
}

// in `ctx`, so its events work with the other queues of the context
static pinned_copy_queue create_pinned_copy_queue(const sycl::device &dev,
                                                  const sycl::context &ctx,
                                                  copy_engine_pick pick) {
    pinned_copy_queue res;
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    uint32_t group_count = 0;
    ze_result_t query_result =
        zeDeviceGetCommandQueueGroupProperties(ze_dev, &group_count, nullptr);
    if (query_result != ZE_RESULT_SUCCESS) {
        LOG_WARN("zeDeviceGetCommandQueueGroupProperties failed: ",
                 ccl::ze::to_string(query_result));
        res.fallback_reason = "queue_group_query_failed";
        return res;
    }
    std::vector<ze_command_queue_group_properties_t> groups(
        group_count, { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES });
    query_result = zeDeviceGetCommandQueueGroupProperties(ze_dev, &group_count, groups.data());
    if (query_result != ZE_RESULT_SUCCESS) {
        LOG_WARN("zeDeviceGetCommandQueueGroupProperties failed: ",
                 ccl::ze::to_string(query_result));
        res.fallback_reason = "queue_group_query_failed";
        return res;
    }

    uint32_t num_queues = 0;
    for (uint32_t i = 0; i < group_count; i++) {
        const bool is_copy = groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        const bool is_compute = groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        if (is_copy && !is_compute && groups[i].numQueues > num_queues) {
            res.ordinal = i;
            num_queues = groups[i].numQueues;
        }
    }
    // the last engine runs next to the stream queue's copies: one engine would serialize them
    if (num_queues == 0 || (pick == copy_engine_pick::last && num_queues < 2)) {
        res.fallback_reason = num_queues ? "single_copy_engine" : "no_copy_engine_group";
        res.ordinal = -1;
        return res;
    }
    res.index = num_queues - 1;

    ze_command_queue_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    desc.ordinal = res.ordinal;
    desc.index = res.index;
    // explicit only: the driver must not move the list to another engine
    desc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY | ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    ze_command_list_handle_t ze_cmd_list = nullptr;
    ze_result_t result = zeCommandListCreateImmediate(ze_ctx, ze_dev, &desc, &ze_cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        LOG_WARN("zeCommandListCreateImmediate (",
                 res.ordinal,
                 ",",
                 res.index,
                 ") failed: ",
                 ccl::ze::to_string(result));
        res.fallback_reason = "cmdlist_create_failed";
        return res;
    }

    // same context as q, so events can be shared between the two queues
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> queue_input{
        ze_cmd_list,
        dev,
        sycl::ext::oneapi::level_zero::ownership::transfer,
        sycl::property_list{ sycl::property::queue::in_order{} }
    };
    try {
        res.queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(queue_input, ctx);
    }
    catch (const std::exception &e) {
        // make_queue did not take ownership of the list: destroy it; the caller keeps its copies
        // on the stream queue
        LOG_WARN("make_queue for copy engine (",
                 res.ordinal,
                 ",",
                 res.index,
                 ") failed: ",
                 e.what());
        zeCommandListDestroy(ze_cmd_list);
        res.fallback_reason = "make_queue_failed";
        return res;
    }
    LOG_DEBUG("created copy engine queue: ordinal ", res.ordinal, ", index ", res.index);
    return res;
}

const pinned_copy_queue &get_stream_copy_engine_queue(ccl_stream *stream,
                                                      std::shared_ptr<pinned_copy_queue> &slot,
                                                      copy_engine_pick pick) {
    // not locked: a stream is driven by one thread at a time
    if (!slot) {
        sycl::queue q = stream->get_native_stream();
        sycl::device dev = q.get_device();
        if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) {
            pinned_copy_queue res;
            res.fallback_reason = "not_level_zero";
            slot = std::make_shared<pinned_copy_queue>(std::move(res));
        }
        else {
            slot = std::make_shared<pinned_copy_queue>(
                create_pinned_copy_queue(dev, q.get_context(), pick));
        }
    }
    return *slot;
}

// queue on the last engine of the copy-only group (CRI: ordinal 1, index 1), in the context of
// stream's queue. The driver offloads copies on that queue to the first engine of the group, so
// the two can overlap. Requires a copy-only group with >= 2 engines. Made on first use and kept
// in ccl_stream::self_copy_queue, so streams do not order each other's self copies.
static const pinned_copy_queue &get_last_copy_engine_queue(ccl_stream *stream) {
    return get_stream_copy_engine_queue(stream, stream->self_copy_queue, copy_engine_pick::last);
}

std::optional<sycl::queue> get_self_copy_queue(ccl_stream *stream,
                                               const char *collective,
                                               size_t self_bytes,
                                               size_t min_bytes,
                                               const char *fallback_reason) {
    if (!is_cri_card(stream->get_device_family())) {
        return std::nullopt;
    }

    // the stream's cached entry, only looked up when nothing below rules the second engine out
    const pinned_copy_queue *self_q = nullptr;
    if (fallback_reason) {
        // keep the caller's reason
    }
    else if (min_bytes == 0) {
        // CCL_SYCL_<COLL>_SELF_COPY_THRESHOLD=0
        fallback_reason = "disabled";
    }
    else if (self_bytes < min_bytes) {
        // below the threshold the cross-queue dependency and join cost more than the overlap saves
        fallback_reason = "small_message";
    }
    else if (sycl::queue q = stream->get_native_stream(); use_recording_path(q)) {
        // graph recording only captures stream's queue
        fallback_reason = "recording_path";
    }
    else if (ccl::global_data::get().shared_data->is_multi_thread_instance) {
        // cross-queue dependencies can hang with multi-thread communicators
        fallback_reason = "multi_thread";
    }
    else {
        self_q = &get_last_copy_engine_queue(stream);
        fallback_reason = self_q->fallback_reason;
    }

    // remote copies stay on stream's queue, the driver picks their engine
    const bool fallback = !self_q || !self_q->queue.has_value();
    LOG_DEBUG("collective=",
              collective,
              " selected_path=",
              fallback ? "serial_ce" : "parallel_ce",
              " self_engine_group=",
              fallback ? "stream" : std::to_string(self_q->ordinal),
              " self_queue_index=",
              fallback ? "driver_selected" : std::to_string(self_q->index),
              " fallback=",
              fallback ? "true" : "false",
              " fallback_reason=",
              fallback ? fallback_reason : "none",
              " self_bytes=",
              self_bytes,
              " threshold=",
              min_bytes);
    if (fallback) {
        return std::nullopt;
    }
    return self_q->queue;
}

sycl::queue &get_default_queue() {
    static sycl::queue default_queue;
    return default_queue;
}

void copy_data(const int dsize,
               const int N,
               std::array<void *, MAX_GPUS> dst,
               std::array<void *, MAX_GPUS> src,
               const size_t count,
               sycl::queue q,
               std::vector<sycl::event> deps,
               std::vector<sycl::event> &out) {
    for (int i = 0; i < N; i++) {
        sycl::event e = q.submit([=](sycl::handler &h) {
            h.depends_on(deps);
            h.memcpy(dst[i], src[i], dsize * count);
        });
        out.push_back(e);
    }
}

sycl::event sycl_average(sycl::queue &q,
                         void *reduce_buf,
                         const size_t reduce_count,
                         const size_t total_ranks,
                         ccl::datatype dtype,
                         std::vector<sycl::event> &dep_events) {
    auto lambda = [&]<typename T>() {
        const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();
        bool use_full_vector = can_use_full_vector(reduce_buf, reduce_buf, reduce_count, dsize);
        if (use_full_vector) {
            constexpr int vec_size = get_num_elements<T, 8, true>();
            return reduce_average_invoke<T, vec_size, 16>(
                q, reduce_buf, reduce_count, total_ranks, dep_events);
        }
        else {
            constexpr int vec_size = get_num_elements<T, 8, false>();
            return reduce_average_invoke<T, vec_size, 64>(
                q, reduce_buf, reduce_count, total_ranks, dep_events);
        }
    };
    return invoke_scaleout_collective(lambda, dtype);
}

bool check_mpi_supports_rdma() {
    if (ccl::global_data::env().atl_transport != ccl_atl_mpi) {
        // ofi collective only supports host memory
        return false;
    }
#ifdef CCL_ENABLE_MPI
    auto lib_attr = atl_mpi_ctx::get_lib_attr();
    if (lib_attr.type == atl_mpi_ctx::ATL_MPI_LIB_IMPI && lib_attr.hmem == 1) {
        const char *env_val = getenv("I_MPI_OFFLOAD");
        int offload = 0;
        if (env_val != nullptr)
            offload = atoi(env_val);

        if (offload == 0) {
            LOG_INFO("Intel MPI does not support GPU RDMA");
            return false;
        }
        return true;
    }
    else if (lib_attr.type == atl_mpi_ctx::ATL_MPI_LIB_MPICH && lib_attr.hmem == 1) {
        const char *env_val = getenv("MPIR_CVAR_CH4_OFI_ENABLE_HMEM");
        int gpu_rdma = 0;
        if (env_val != nullptr)
            gpu_rdma = atoi(env_val);

        env_val = getenv("MPIR_CVAR_CH4_OFI_ENABLE_GPU_PIPELINE");
        int gpu_pipeline = 0;
        if (env_val != nullptr)
            gpu_pipeline = atoi(env_val);

        if (!gpu_rdma && !gpu_pipeline) {
            LOG_INFO("MPICH does not support GPU RDMA");
            return false;
        }
        return true;
    }
    return false;
#else
    return false;
#endif
}

sycl::event pt2pt_pre_sync(sycl::queue &q,
                           const std::vector<sycl::event> &deps,
                           ccl_comm *comm,
                           bool do_send,
                           int peer_rank,
                           uint64_t tag) {
    auto init_fn = [=](atl_req_t &req) -> bool {
        int ep_idx = 0;
        char data[1] = { 1 };
        size_t n = sizeof(data);
        auto atl = comm->get_atl_comm();

        if (do_send) {
            ATL_CALL_THROW_IF_ERROR(atl->send(ep_idx, data, n, peer_rank, tag, req));
            LOG_DEBUG("pt2pt pre-sync SEND init, tag=", tag);
        }
        else {
            LOG_DEBUG("pt2pt pre-sync RECV init, tag=", tag);
            ATL_CALL_THROW_IF_ERROR(atl->recv(ep_idx, data, n, peer_rank, tag, req));
            if (group_impl::is_group_active()) {
                ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
            }
        }

        if (!group_impl::is_group_active()) {
            ATL_CALL_THROW_IF_ERROR(atl->check(ep_idx, req));
            if (!req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
            }
        }
        return true;
    };

    // wait_fn: performs the blocking wait
    auto wait_fn = [=](atl_req_t &req) {
        if (req.is_completed) {
            return;
        }
        int ep_idx = 0;
        auto atl = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
        LOG_DEBUG("pt2pt pre-sync wait done, tag=", tag);
    };

    if (group_impl::is_group_active()) {
        return ccl_host_task(q, deps, [=]() {
            atl_req_t req{};
            init_fn(req);
        });
    }

    atl_req_t req{};
    sycl::event e = ccl_host_task(q, deps, [=, &req]() {
        init_fn(req);
    });

    // Wait for the host_task to post the send/recv
    e.wait();
    // then do the final blocking wait here
    wait_fn(req);
    return e;
}

sycl::event post_host_task_ack(sycl::queue &q,
                               const std::vector<sycl::event> &deps,
                               ccl_comm *comm,
                               bool do_send,
                               int peer_rank,
                               uint64_t ack_tag) {
    auto ack_data = std::make_shared<char>(1);
    auto ack_driver = [=](atl_req_t &req, bool sync, bool init) -> bool {
        int ep_idx = 0;
        char *data = ack_data.get();
        size_t n = 1;
        auto atl = comm->get_atl_comm();
        if (req.is_completed) {
            return true;
        }
        if (init) {
            if (do_send)
                ATL_CALL_THROW_IF_ERROR(atl->send(ep_idx, data, n, peer_rank, ack_tag, req));
            else
                ATL_CALL_THROW_IF_ERROR(atl->recv(ep_idx, data, n, peer_rank, ack_tag, req));
        }

        ATL_CALL_THROW_IF_ERROR(atl->check(ep_idx, req));

        if (!sync) {
            return req.is_completed;
        }

        if (!req.is_completed) {
            ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
        }

        LOG_DEBUG("post_host_task_ack: ", (do_send ? "send" : "recv"), ", tag=", ack_tag);
        return true;
    };
    if (group_impl::is_group_active()) {
        group_impl::set_sycl_queue(q);
        sycl::event e = submit_wait_on_events(q, deps);
        e.wait();
        // post the ack right away, the peer may wait for it before its own
        // group end; only the completion check stays in post-processing
        auto preq = std::make_shared<atl_req_t>();
        ack_driver(*preq, false, /*init=*/true);
        group_impl::add_post_processing_step(
            [=](atl_req_t &, bool sync, bool) -> bool {
                return ack_driver(*preq, sync, /*init=*/false);
            });
        return e;
    }

    atl_req_t req{};
    sycl::event e = ccl_host_task(q, deps, [=, &req]() {
        ack_driver(req, true, /*init=*/true);
    });
    e.wait();
    ack_driver(req, true, /*init=*/false);
    return e;
}
