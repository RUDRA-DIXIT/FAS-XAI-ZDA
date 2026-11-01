"""
schema.py - static facts about the NSL-KDD file format.

Two kinds of constants live here, and they are kept visibly separate:

1. FORMAT facts (column names/order). These come from the public NSL-KDD
   format. Phase 1 *verifies* them against the real files.
2. SPEC_EXPECTED values. These are what the paper / specification SAYS the
   data should look like. They are expectations to be checked, never truth.
   If the real files disagree, the audit reports a mismatch; it does not
   quietly "fix" the data or the expectation.
"""

# --------------------------------------------------------------------------
# 1. File format: 43 comma-separated fields per line, NO header row.
#    41 traffic features + attack label + difficulty score.
# --------------------------------------------------------------------------
RAW_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
LABEL_COL = "label"
# The 43rd column is the dataset's "difficulty score". The paper calls it
# `level` (Figs. 7 and 9, Table 3), so we use the paper's name.
LEVEL_COL = "level"
RAW_COLUMNS = RAW_FEATURES + [LABEL_COL, LEVEL_COL]
N_RAW_COLUMNS = 43
assert len(RAW_FEATURES) == 41 and len(RAW_COLUMNS) == N_RAW_COLUMNS

CATEGORICAL = ["protocol_type", "service", "flag"]
NORMAL_LABEL = "normal"

# Columns we expect to behave in a certain way (checked, not assumed):
RATE_COLUMNS = [c for c in RAW_FEATURES if c.endswith("_rate")]  # expect [0,1]
BINARY_LIKE_COLUMNS = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]  # expect {0,1}; the audit reports any other value it finds

# --------------------------------------------------------------------------
# 2. What the SPEC says (Sec. 2.1 of the specification; paper Figs. 3-5, 7, 9)
# --------------------------------------------------------------------------
# Spec Sec. 2.1: the 41 model inputs, in this order. Names were read from
# paper Figs. 7 and 9 [spec label: I = inferred]. This is typed out in full on
# purpose, so it is an INDEPENDENT check against what the data contains.
CANONICAL_INPUTS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "level",
]
assert len(CANONICAL_INPUTS) == 41 and len(set(CANONICAL_INPUTS)) == 41

# Spec: num_outbound_cmds is absent from Figs. 7/9 [I]. The audit checks
# whether it is a constant column (which would justify leaving it out).
EXPECTED_CONSTANT_COLUMNS = ["num_outbound_cmds"]

SPEC_EXPECTED = {
    # Paper Sec. 2.3 / spec Sec. 2.1 [E]
    "train_rows": 125_973,
    "train_normal": 67_343,
    "train_attack": 58_630,
    "train_attack_classes": 22,
    # Paper Fig. 5: the 14 "known" attack classes [E]
    "known_attacks": [
        "neptune", "satan", "ipsweep", "portsweep", "smurf", "nmap", "back",
        "teardrop", "warezclient", "pod", "guess_passwd", "buffer_overflow",
        "warezmaster", "land",
    ],
    # Paper Fig. 4: the 8 zero-day (rare) classes and their counts [E]
    "zero_day_counts": {
        "imap": 11, "rootkit": 10, "loadmodule": 9, "ftp_write": 8,
        "multihop": 7, "phf": 4, "perl": 3, "spy": 2,
    },
    # Paper Sec. 2.3: other counts quoted in the text [E]
    "other_counts": {
        "guess_passwd": 53, "buffer_overflow": 30,
        "warezmaster": 20, "land": 18,
    },
    # Spec Alg. 2 / D-12: rarity threshold. "fewer than 15" => n <= 14.
    "delta": 14,
    # Paper Fig. 3 (partition counts) [E]
    "fig3": {
        "normal": {"train": 40_405, "val": 13_469, "test": 13_469},
        "attack": {"train": 33_558, "val": 11_715, "test": 11_769},
    },
    "split_ratio": (0.6, 0.2, 0.2),
}
