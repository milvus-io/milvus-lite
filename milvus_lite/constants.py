# ── MemTable ──
MEMTABLE_SIZE_LIMIT = 10_000

# ── Compaction ──
MAX_DATA_FILES = 32
COMPACTION_MIN_FILES_PER_BUCKET = 4
COMPACTION_BUCKET_BOUNDARIES = [1_000_000, 10_000_000, 100_000_000]  # bytes

# Max rows a single post-compaction segment may hold. Compaction skips
# merge groups whose combined row count would exceed this limit. Keeps
# per-segment index build cost bounded — critical for HNSW_SQ / HNSW_PQ
# where build time is super-linear in segment size.
MAX_SEGMENT_ROWS = 100_000

# ── File naming ──
SEQ_FORMAT_WIDTH = 6
DATA_FILE_TEMPLATE = "data_{min:0{w}d}_{max:0{w}d}.parquet"
DELTA_FILE_TEMPLATE = "delta_{min:0{w}d}_{max:0{w}d}.parquet"
WAL_DATA_TEMPLATE = "wal_data_{n:0{w}d}.arrow"
WAL_DELTA_TEMPLATE = "wal_delta_{n:0{w}d}.arrow"

# ── Partition ──
DEFAULT_PARTITION = "_default"
ALL_PARTITIONS = "_all"

# ── Partition Key ──
DEFAULT_NUM_PARTITIONS = 16
PARTITION_KEY_BUCKET_PREFIX = "_pk_"

# ── Filter expression cache ──
# Per-Collection LRU cache for compiled filter expressions. Phase F2c.
# Most search workloads reuse a small set of expression strings — 256
# is plenty for typical use and bounds memory at ~hundreds of KB.
FILTER_CACHE_SIZE = 256
