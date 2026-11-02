# Tuning oneCCL on CRI with a selection policy file

This guide explains how to measure the best settings for oneCCL collectives on a CRI node, turn them into a
`CCL_SELECTION_POLICY_PATH` JSON file, and use and check that file. It uses
`deps/libccl/policies/cri_policy.json` from the oneCCL repository as the worked example.
Commands use two placeholders: `<repo>` for the oneCCL repository root and `<build>` for the oneCCL build
directory.

No tooling is assumed beyond oneCCL, its benchmarks and MPI.

## What you need

- **oneCCL** with the `copy_path`, `simple_chunk_size` and `simple_ring_segment_size` knobs in its
  selection-policy knob registry, built with its tests. A build without them loads the file but warns
  `unknown knob '<name>'` for each knob it does not know, and ignores it. To check a build, run any
  benchmark with `CCL_LOG_LEVEL=info` and the file: there must be no `unknown knob` warning.
- **The oneCCL benchmarks**, built from the repository's `tests/benchmark` into `<build>/tests/benchmark`.
- **An MPI launcher**, Intel® MPI Library or Open MPI.
- **Optional:** `ze_peer` from level-zero-tests to measure GPU-to-GPU bandwidth.

## 1. What a policy file can change

oneCCL reads the file once, when `CCL_SELECTION_POLICY_PATH` points to it. For every large scale-up call of
allreduce, reduce_scatter, allgather(v), alltoall and alltoallv, it looks up the band that matches the
collective, the rank count and the message size, and applies that band's settings to that call only.
Calls made at the same time from other threads keep their own settings.

| Setting in the file | What it sets | Collectives | Scope |
|---|---|---|---|
| `copy_path=copy_engine` / `kernel` / `default` | `CCL_SYCL_COPY_ENGINE` (alltoall, alltoallv: `CCL_SYCL_ALLTOALL_COPY_ENGINE`) | all five | per band |
| `simple_chunk_size=<bytes>` | `CCL_SYCL_<COLL>_SIMPLE_CHUNK_SIZE`, and for alltoall on `is_arc=yes` devices such as CRI `CCL_SYCL_ALLTOALL_CHUNKING_THRESHOLD` | allreduce, reduce_scatter, allgatherv, alltoall | per band |
| `simple_ring_segment_size=<bytes>` | `CCL_SYCL_<COLL>_SIMPLE_RING_SEGMENT_SIZE` | allreduce, reduce_scatter, allgatherv (tmp-buffer path only) | per band |
| `global_config` entry `"CCL_SYCL_TMP_BUF_SIZE"` | `CCL_SYCL_TMP_BUF_SIZE` (the tmp buffer allocation); any other `CCL_*` variable works the same way | all | process-wide |

A knob sits inside a band's `algorithm=` node and holds for that band only.

`<COLL>` is `ALLREDUCE`, `REDUCE_SCATTER`, `ALLGATHERV` or `ALLTOALL`.

### Precedence

| Setting | Wins, highest first |
|---|---|
| Knobs in the tree (`copy_path`, `simple_chunk_size`, `simple_ring_segment_size`) | the knob that holds for the call's band, then the environment variable, then the built-in default. A band without the knob keeps the variable's value. For `simple_chunk_size` the variable is the `CCL_SYCL_<COLL>_SIMPLE_CHUNK_SIZE` spec; for alltoall on `is_arc=yes` devices such as CRI, the knob also replaces `CCL_SYCL_ALLTOALL_CHUNKING_THRESHOLD`. |
| `global_config` entries | the file, then the environment variable, then the built-in default. |

### Where the file has no effect

- **Small messages.** On CRI, messages up to the "simple threshold" run the low-latency (LL) kernels, which
  ignore every knob. The thresholds compare the size each collective passes to the policy query: allreduce
  4 MB (whole buffer), reduce_scatter 8 MB (whole buffer), allgatherv 1 MB per rank, alltoall 512 KB per peer.
- **Broadcast, scatter and reduce.** This guide was tested only on the five collectives above. Broadcast,
  scatter and reduce also read the file, but the CRI file has no sections for them and the guide does not
  cover tuning them.
- **Multi-thread communicators and GPUs without P2P access.** The CRI file keeps the built-in LL rules for
  them, which come before its bands.
- **Rank counts without a branch.** A `comm_size` the file does not list gets the built-in choice from the
  final `msg_size=any` rule.
- **Graph recording.** Recorded calls consult the file like other calls. In the CRI file the bands sit under
  `is_recording=no`, so recorded calls get the built-in choice and no knobs.
- **Multi-node runs.** The file tunes the scale-up phase only.

## 2. Before you start

1. **Record the environment.** Write down the oneCCL commit, the NEO, IGC and Level Zero loader versions,
   the kernel and xe driver, and the GPU firmware, and keep them with the results. Re-tune when any of them
   changes: a driver update can change the right choice for several bands.
2. **Learn the topology.** Find out which GPUs share a PCIe switch and which CPU socket each GPU attaches
   to, for example from `lspci -t` and each device's `numa_node` in sysfs. Measure GPU-to-GPU bandwidth per
   link type with `ze_peer`. Bandwidth across sockets is lower than within a socket, so tune small rank
   counts on GPUs of one socket. A run that spans both sockets is limited by the link between them.
3. **Pin ranks to the GPU's socket.** Bind each rank to a CPU on the same NUMA node as its GPU (`lscpu`
   lists the CPUs of each node). For runs that span both sockets, split the ranks the same way.
4. **Make sure the GPUs are idle.** If others share the machine, check which processes hold each GPU
   before every series, for example by looking for open `/dev/dri/renderD*` handles in `/proc/*/fd`. Also
   look for your own benchmark ranks left over from interrupted runs, which keep holding the GPUs.

## 3. The file, field by field

Excerpt from `cri_policy.json`, reduce_scatter, with only the 4-rank branch shown:

```json
{
  "schema_version": 1,
  "notes": { ... },
  "collective=reduce_scatter": {
    "p2p_domain=any": {
      "phase=scaleup": {
        "is_arc=yes": {
          "is_multi_thread=yes": { "protocol=sycl_ll": { "algorithm=ring": { "ll_table=oneshot:0-256;ring:257-max": {} } } },
          "has_p2p_access=no": { "protocol=sycl_ll": { "algorithm=ring": { "ll_table=oneshot:0-256;ring:257-max": {} } } },
          "send_msg_size<=8388608": { "protocol=sycl_ll": { "algorithm=ring": { "ll_table=oneshot:0-256;ring:257-max": {} } } },
          "is_recording=no": {
            "comm_size=4": {
              "msg_size<4194304": { "protocol=sycl_simple": { "algorithm=large": {} } },
              "msg_size<=8388608": { "protocol=sycl_simple": { "algorithm=large": { "copy_path=kernel": {} } } },
              "msg_size<=16777216": { "protocol=sycl_simple": { "algorithm=large": { "copy_path=kernel": {}, "simple_chunk_size=4Mb": {} } } },
              "msg_size<=33554432": { "protocol=sycl_simple": { "algorithm=large": { "copy_path=copy_engine": {} } } },
              "msg_size=any": { "protocol=sycl_simple": { "algorithm=large": { "copy_path=kernel": {}, "simple_ring_segment_size=4Mb": {} } } }
            }
          },
          "msg_size=any": { "protocol=sycl_simple": { "algorithm=large": {} } }
        },
        "is_arc=no": {
          "send_msg_size<=2097152": { "protocol=sycl_simple": { "algorithm=small": {} } },
          "msg_size=any": { "protocol=sycl_simple": { "algorithm=large": {} } }
        }
      }
    }
  }
}
```

| Key | Meaning | How to choose it |
|---|---|---|
| `schema_version` | Always `1`. | Fixed. |
| `global_config` | `"CCL_<VAR>": "<value>"` entries that replace oneCCL variables, including ones set in the environment, read once at startup. The CRI file has none. | See 4.5. |
| `notes` | Free text. The engine skips it with a one-line warning, as it does any other unknown top-level key. | Record the source data, environment and decisions here. |
| keys starting with `_` | Comments anywhere in the tree, skipped by the engine. | Use `_note`, `_sizes`, `_adjusted` to explain a band. |
| `collective=<name>` | `allreduce`, `reduce_scatter`, `allgather` (also serves allgatherv), `alltoall`, `alltoallv`. A missing collective keeps its defaults. | One section per tuned collective. |
| `p2p_domain=any`, `phase=scaleup`, `is_arc=yes` / `is_arc=no` | Each collective's scale-up tree from `collectives_policy.json`, which reproduces the built-in defaults: LL for multi-thread communicators, missing P2P access and small messages, `sycl_simple` / `large` otherwise. | Copy it unchanged and add the bands on the `is_arc=yes` side, before its final `msg_size=any` rule. |
| `is_recording=no` | Holds the bands, so graph-recorded calls skip them. | Keep as is. |
| `comm_size=<N>` | Matches the communicator size exactly. | One branch per rank count you measured. Unlisted sizes get the final built-in rule. |
| `msg_size<X`, `msg_size<=X`, `msg_size=any` | Size bands, tried in file order, first match wins. `<=X` therefore means "above the previous bound, up to X". `any` catches everything left. | Put one band per run of sizes with the same best setting, see 4.6. |
| `protocol=sycl_simple`, `algorithm=large` | The decision: the default kernel's large path, the built-in choice at these sizes. The knobs sit inside it; without knobs the band keeps the defaults. | Use it for every band, with knobs or without. A band without a decision falls through to the next band. |
| `copy_path=...` | Copy engine or compute kernel for the peer copies of this call. `default` = no override. | The larger of the copy-engine and best-kernel bandwidth, see 4.2. |
| `simple_chunk_size=<bytes>` | The dispatcher splits the call into pieces of at most this size. `0` is the same as no knob. | Sweep, see 4.2. Only on kernel bands in this file. |
| `simple_ring_segment_size=<bytes>` | Bytes per pipeline slot of the ring kernel. | Sweep, see 4.2. Must fit the allocation, see 4.5. |

**What `msg_size` measures.** It is per-rank bytes, so it is not the benchmark's Size column for most
collectives:

| Collective | `msg_size` | From the benchmark Size |
|---|---|---|
| allreduce | count × dtype size, the whole buffer | Size |
| reduce_scatter | receive count × dtype size | Size / ranks |
| allgather(v) | send count × dtype size | Size / ranks |
| alltoall | per-peer count × dtype size | Size / ranks |
| alltoallv | largest count × dtype size | largest per-peer block |

Sizes in knob values and `global_config` accept `Kb`, `Mb`, `Gb` (×1024): write `4Mb` for 4 MiB, because `K`,
`M`, `G` mean ×1000 (`4M` = 4000000). Filter values in the file are plain byte counts. The engine also offers `dtype=<name>` (for example `float32`,
`bfloat16`) and `pow2` filters, which the current file does not use. The current file was tuned with fp32;
if another data type behaves differently, add a `dtype=` branch for it.

## 4. Measuring, 16 MB to 1 GB

Measure every collective at every rank count you will support, from 16 MB to 1 GB total in factors of 2,
both out-of-place and in-place. Use the same benchmark arguments for every measurement.

### The benchmark

Use the oneCCL benchmarks, which follow the nccl-tests conventions: `allreduce_perf`, `reduce_scatter_perf`,
`allgather_perf`, `alltoall_perf` and `alltoallv_perf` in `<build>/tests/benchmark`. Run one MPI rank per GPU
and select the GPUs with `ZE_AFFINITY_MASK`:

```
ZE_AFFINITY_MASK=0,1,2,3 mpirun -n 4 <build>/tests/benchmark/allreduce_perf -g 1 -b 16M -e 1G -f 2 -d fp32 -c 1 -n 50 -w 20
```

With Open MPI, oneCCL cannot read the local rank from the launcher. Set `CCL_PROCESS_LAUNCHER=none` and give
each rank `CCL_LOCAL_RANK` and `CCL_LOCAL_SIZE` from `OMPI_COMM_WORLD_LOCAL_RANK` and
`OMPI_COMM_WORLD_LOCAL_SIZE`. A one-line wrapper per rank does it:

```
ZE_AFFINITY_MASK=0,1,2,3 CCL_PROCESS_LAUNCHER=none mpirun -n 4 \
  sh -c 'CCL_LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK CCL_LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE exec "$0" "$@"' \
  <build>/tests/benchmark/allreduce_perf -g 1 -b 16M -e 1G -f 2 -d fp32 -c 1 -n 50 -w 20
```

The Intel® MPI Library needs neither. Add the launcher's binding options to pin ranks as described in section 2.

| Option | Meaning | Use |
|---|---|---|
| `-g` | GPUs per process | `1` |
| `-b`, `-e` | Smallest and largest message size, `K`/`M`/`G` accepted | `16M`, `1G` |
| `-f` | Size step factor (`-i` for a fixed step instead) | `2` |
| `-d` | Data type | `fp32`, or the type you will run |
| `-n`, `-w` | Timed and warm-up iterations | `50` and `20` for sweeps, at least `20` and `5` for checks |
| `-c` | Check the results | `1` |
| `-p` | Also run in-place | `1`, the default |
| `-z` | Blocking collectives | leave at `0` |

How to read the output:

- **Data rows** are the lines whose first two fields are integers: size, count, type, reduction, root, then
  time, algbw, busbw and `#wrong` for out-of-place, then the same four for in-place.
- **Size** is the whole buffer, as in nccl-tests. Convert it to the policy's `msg_size` with the table in
  section 3.
- **busbw** is the column to compare.
- **`#wrong`** must be `0` in every row, and the run must end with `Out of bounds values : 0 OK`. In-place
  alltoall shows `N/A`, because oneCCL does not support it and the benchmark does not check it.
- **Measure at the default log level.** At `CCL_LOG_LEVEL=info`, log lines interleave with the data rows.

### 4.1 What a measurement runner must do

You will run a few hundred measurements, so write a small runner. For each measurement it should:

1. take the collective, the rank count, the GPU list and a set of `VAR=VAL` settings for this run only;
2. run the benchmark as above with those settings in the environment, with a timeout;
3. save the full output in its own log file, headed by the date, oneCCL commit, driver versions, GPUs,
   settings and benchmark arguments;
4. parse the data rows and append one line per size to a results table, with the columns rank count,
   collective, a tag naming the settings, repeat number, size, out-of-place busbw, in-place busbw, both
   `#wrong` values, and the exit code;
5. run only one benchmark at a time on the machine;
6. on restart, skip measurements that already completed with exit code 0, and run failed ones again.

### 4.2 The measurement matrix

One variable chooses between the copy engine and the compute kernel, and which one depends on the
collective:

| Collective | Copy-path variable | Default on CRI |
|---|---|---|
| allreduce, reduce_scatter, allgather | `CCL_SYCL_COPY_ENGINE` | `0`, kernel |
| alltoall, alltoallv | `CCL_SYCL_ALLTOALL_COPY_ENGINE` | `-1`, auto: copy engine, see 4.3 |

In the matrix, "copy path = 1" means setting that collective's copy-path variable to `1`, and "copy path = 0"
setting it to `0`.

| Series | Settings | Collectives | Values swept |
|---|---|---|---|
| Defaults | nothing set | allreduce, reduce_scatter, allgather, alltoall | - |
| Copy engine | copy path = 1 | allreduce, reduce_scatter, allgather, alltoall | - |
| Kernel | copy path = 0 | allreduce, reduce_scatter, allgather, alltoall | - |
| Kernel + chunk | copy path = 0, `CCL_SYCL_<COLL>_SIMPLE_CHUNK_SIZE=C` | allreduce, reduce_scatter, allgather, alltoall | coarse: allreduce 2M-256M, others 1M-64M, powers of 2 |
| Kernel + segment | copy path = 0, `CCL_SYCL_<COLL>_SIMPLE_RING_SEGMENT_SIZE=S` | allreduce, reduce_scatter | 1M-512M, powers of 2 |
| Kernel + chunk + segment | copy path = 0, both of the above | allreduce, reduce_scatter | a small grid around the two bests |

Notes on the series:

- **Defaults** are the baseline every choice is judged against.
- **Alltoall uses the copy engine by default on CRI**, so its kernel series must set
  `CCL_SYCL_ALLTOALL_COPY_ENGINE=0` explicitly, and its copy-engine series `=1` to skip the auto rule in
  4.3. The other collectives default to the kernel.
- **Measure the copy engine without chunk or segment settings.** That is how its bands are encoded, and in
  one test adding a 16 MB chunk to allgather's copy engine did not help.
- **Chunk, refine after the coarse pass.** For each size, add 0.75× and 1.5× the best value. If the best
  value is at either end of the range, extend the range there. Skip values at or above the block they would
  split, the whole buffer for allreduce and Size / ranks for the others: they behave the same as no chunk.
- **Segment, skip two kinds of values.** Skip the default segment, 384 MB / max(ranks, 3), and values above
  1 GB / ranks. When a segment times the pipeline depth exceeds 384 MB, also set
  `CCL_SYCL_TMP_BUF_SIZE` to segment × depth for that measurement. Otherwise oneCCL clamps the segment.
- **Do not sweep the segment for allgather.** Its default large path writes to peers directly and ignores
  the segment, which only affects allgather's tmp-buffer path.
- **Repeat** every candidate at least three times, interleaving the series rather than running all repeats
  of one series back to back.

### 4.3 Built-in rules that affect what you measure

- **alltoall in auto mode** (`CCL_SYCL_ALLTOALL_COPY_ENGINE` unset, or a band without `copy_path`) runs
  the kernel at 2 ranks and when the peers share a PCIe switch, up to
  `CCL_SYCL_ALLTOALL_CRI_COPY_ENGINE_THRESHOLD` (4 MB) per peer, and the copy engine otherwise.
  `copy_path=copy_engine` or `kernel`, or the variable at `1` or `0`, skips the rule.
- **reduce_scatter at 2 ranks** switches to the copy engine by itself once a call's per-rank piece reaches
  4 × min(16 MB, segment). Its "kernel" bands therefore run memcpy at most sizes. That is fine as long as
  the measurement and the policy use the same settings.
- **The segment is clamped** to the allocation divided by the pipeline depth, with one warning per process.
  Depth is max(ranks, 3) for allreduce and reduce_scatter, and ranks for allgather.

### 4.4 Rank placement

Measure 2- and 4-rank runs on GPUs of one socket, the way they will run in production. If you will run 4
ranks on either socket, check that both give the same results before using one table for both.

### 4.5 Allocation size

The default tmp buffer is 384 MB. If a chosen segment times the pipeline depth exceeds it, add
`"CCL_SYCL_TMP_BUF_SIZE": "<bytes>"` to `global_config`, at least the largest segment × depth over all bands. The allocation
costs that much device memory per rank, for the whole run, so only raise it when a band measurably needs
it. In earlier tests, a 1.5 GB allocation brought no gain.

### 4.6 Pick the best setting per band

For each collective, rank count, size and mode (out-of-place and in-place), using the median of the
repeats:

1. The candidates are the defaults, the copy engine, and the best kernel setting from the chunk and segment
   series.
2. Take the highest busbw.
3. If the winner is within about 2% of the defaults, keep the defaults. Fewer knobs means fewer
   surprises after a driver update.
4. **Out-of-place and in-place may disagree.** The file cannot tell them apart, so choose one setting per
   size. Pick the option with the smaller worst-case loss across the two modes. If both modes want the
   kernel with different values, take the values from the mode where the kernel's margin over the copy
   engine is larger. Record each such decision in the file's `notes`.
5. Merge neighbouring sizes with the same setting into one band.

## 5. Write the JSON file

Keep the per-band decisions from 4.6 in a table, one row per collective, rank count and size, with the
chosen path and settings. Generate the file from that table rather than editing JSON by hand, so it can be
regenerated when the measurements change. The conversion:

1. **One section per collective**, `collective=<name>`, starting from that collective's
   `p2p_domain=any` / `phase=scaleup` tree in `collectives_policy.json`. Keep `is_arc=no` and the LL rules of
   `is_arc=yes` as they are. Before the final `msg_size=any` rule of `is_arc=yes`, add `is_recording=no` with
   one `comm_size=<N>` branch per rank count.
2. **Convert each benchmark Size to `msg_size`** with the table in section 3.
3. **Bands, in increasing order.** First a guard band below the smallest measured size,
   `"msg_size<X": { "protocol=sycl_simple": { "algorithm=large": {} } }`; sizes small enough for LL never
   reach it. Then one band per run of sizes with the same setting, keyed
   `msg_size<=<largest size of the run>`. The last band is `msg_size=any`, which also covers sizes above the
   largest measured one.
4. **Each band's leaf** is `{ "protocol=sycl_simple": { "algorithm=large": { <knobs> } } }`, with each knob
   written as `"<name>=<value>": {}`:
   - the path as `copy_path=copy_engine` or `copy_path=kernel`;
   - a chunk as `simple_chunk_size=<C>`, sizes in `Kb`/`Mb`/`Gb`;
   - a segment as `simple_ring_segment_size=<S>`. If your data recorded the ring size as
     `CCL_SYCL_TMP_BUF_SIZE=T` instead, the segment is T / depth;
   - a band that keeps the defaults as the same leaf with no knobs.
5. **`global_config`:** `"CCL_SYCL_TMP_BUF_SIZE": "1536Mb"` or similar if 4.5 needs it.
6. **`notes`:** the measurement date, oneCCL commit, driver versions, data source, the conversion rules,
   and every in-place/out-of-place conflict and how it was resolved.

Then check the file:

- It must parse as JSON.
- **Cross-check it against the table** with a separate small evaluator that reads the file the way oneCCL
  does: walk each level in file order, skip keys starting with `_`, take the first key whose filter matches
  (`is_arc`, `is_multi_thread`, `has_p2p_access`, `is_recording`; `comm_size=N`; `msg_size` and
  `send_msg_size` with `<X`, `<=X`, `=any`), and stop at the first `algorithm=` key. The knobs are the keys
  under it whose value is `{}`. For every measured row, the evaluator's knobs must equal the chosen
  setting, or the row must be a recorded conflict.

## 6. Use the file and check it

### Use it

Point oneCCL at the file with an absolute path before the communicator is created:

```
export CCL_SELECTION_POLICY_PATH=<repo>/deps/libccl/policies/cri_policy.json
```

Run with the same environment the measurements used: driver, MPI and rank setup, pinning, and any
variables you set.

### Check what oneCCL selected

Run once with `CCL_LOG_LEVEL=debug` and look at rank 0's lines. The decision line of each call is printed
only at debug level; the other lines below are also printed at `info`.

| Log line | Meaning |
|---|---|
| `selection policy: loaded N collective(s) from '<path>'` | The file was read. |
| `scaleup json: <coll> knobs: copy_path=... simple_chunk_size=...` | The knobs applied to a call, printed just before its decision line. Calls in a band without knobs print nothing. |
| `scaleup <coll>[json] protocol=... algo=... msg_bytes=N count=... send_bytes=S is_arc=... is_multi_thread=... has_p2p_access=... is_recording=...` | The decision for a call, with the inputs the file filters on. `[json]` means it came from the file; without it, no rule matched and the built-in selection ran. With an `ll_table` knob, LL prints `algo=unspecified`: the table picks the algorithm per chunk. |
| `selection policy: CCL_SYCL_TMP_BUF_SIZE=<value> from global_config` | A `global_config` entry was used, also when the variable is set in the environment. |
| `simple ring segment size N exceeds CCL_SYCL_TMP_BUF_SIZE / pipeline depth ..., clamping to M` | A segment did not fit the allocation, see 4.5. |
| `selection policy: ... unknown filter`, `unknown knob`, `invalid value ... for knob` | A mistake in the file. That entry is ignored. |

The warning `ignoring unknown top-level key 'notes'` is expected.

For every benchmark size, take the decision line and the `knobs:` line before it, and compare the protocol,
algorithm and knobs with what the file should give for that collective, rank count, `msg_size` and
`send_msg_size`, for example with the evaluator from section 5. Every call must be a `[json]` decision.
Each size must land in the intended band. This matters most at band edges, where `<` and `<=` differ.

Do not take bandwidth from info- or debug-level runs.

### Check the performance

Run every collective and rank count with the file and without it, back to back on the same GPUs, with the
same arguments and at least two repeats. If the policy is slower than the defaults by more than 2 GB/s in
one mode and not faster in the other, remove that band's knobs (keep the `sycl_simple` / `large` leaf),
add an `_adjusted` note, and
change the table too, so a regeneration keeps the fix. Repeat a band three times before changing it:
single points at 16 MB are often outliers.

## 7. Easy to miss

- **Bandwidth is not portable across drivers.** Keep the driver and firmware versions with every
  table, and re-measure the copy-engine series after a NEO update.
- **The copy-engine series must use the same engine as `copy_path`.** `copy_path` uses SYCL memcpy
  (`CCL_SYCL_COPY_ENGINE`). The Level Zero copy-engine lists (`CCL_ENABLE_ZE_CE`) are a different path and
  cannot be selected from the file.
- **Write knob sizes with `Mb`, not `M`.** `simple_chunk_size=4Mb` and `simple_ring_segment_size=4Mb` are
  4194304 bytes, but `4M` is 4000000, 4.6% less than the size you measured (`K` 2.3%, `G` 6.9%). The engine
  accepts both without a warning. The segment is still rounded down to 16 bytes and the chunk to whole
  elements, so a missing `b` changes the size, not the alignment. Section 3 has the suffix rules.
- **Hand edits to the JSON are lost on regeneration.** Make every change in the table, or in the
  generator's global settings.
- **alltoallv** has its own section key, `collective=alltoallv`. Without it, alltoallv keeps its defaults.
- **Benchmark sizes land exactly on band edges.** `<` and `<=` differ at exactly those sizes. Check the
  decision lines, as in section 6.
- **Multi-threaded (SPMD) communicators** take the built-in LL rule before the bands. Bands added for them
  get no knobs: once a multi-thread communicator exists in the process, oneCCL ignores every knob.
- **Rank counts you did not measure** have no branch and get the built-in choice from the final
  `msg_size=any` rule.
