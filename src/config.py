"""Central configuration for the hybrid edge-cloud tourist-safety project.

Every constant used by the generator, training, simulation and demo lives here,
so the whole study can be re-run with different assumptions by editing one file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
for _d in (DATA_DIR, MODEL_DIR, RESULT_DIR, FIG_DIR):
    _d.mkdir(exist_ok=True)

SEED = 42

# ---------------------------------------------------------------- classes
CLASSES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
# Class mix reported for STSMIRS [1]
CLASS_MIX = [0.171, 0.470, 0.267, 0.092]

# ---------------------------------------------------------------- dataset sizes
N_TOTAL = 20_000
N_TRAIN, N_VAL, N_TEST = 12_000, 4_000, 4_000

# ---------------------------------------------------------------- generator
GEN = dict(
    disp_log_mean=5.7,        # log(metres): median displacement ~300 m
    disp_log_sd=1.0,
    idle_min=15.0,            # Stage-1 gate: idle >= 15 min
    idle_gamma_k=2.0,
    idle_gamma_theta=20.0,    # minutes
    hazard_prev=0.25,
    incident_prev=0.10,
    # latent-risk coefficients
    b_disp=1.00,
    b_night=1.10,
    b_hazard=0.85,
    b_idle=0.15,
    b_context=0.22,           # cloud-only context score z
    b_incident=1.40,          # cloud-only nearby-incident flag
    noise_sd=0.12,            # irreducible noise
    # what the phone actually observes
    gps_log_sd=0.15,          # multiplicative GPS / map-matching error
    gps_floor_m=20.0,
    hazard_flag_error=0.15,   # cached gazetteer flag wrong 15% of the time
)

FEATURES = ["log_disp", "log_idle", "hour_sin", "hour_cos", "hazard_flag"]

# ---------------------------------------------------------------- latency model (seconds)
CLOUD_LATENCY_MEAN = 6.8      # reported in [1]
CLOUD_LATENCY_CV = 0.30       # assumption
FIXED_STAGES = 5.9            # detection 0.8 + DB write 2.1 + DB update 1.7 + notify 1.3, from [1]
CLOUD_ONLY_TOTAL = CLOUD_LATENCY_MEAN + FIXED_STAGES   # 12.7 s

# ---------------------------------------------------------------- cloud model
CLOUD_TARGET_ACC = 0.896      # reported in [1]

# ---------------------------------------------------------------- cost-sensitive rule
# C(true i, pred j) = (i-j)*W[i] for under-prediction, (j-i) for over-prediction
UNDER_WEIGHTS = [1, 1, 3, 10]

# ---------------------------------------------------------------- escalation sweep
THETA_GRID = [round(0.025 * k, 3) for k in range(41)]
LATENCY_FIRST_MAX_ESC = 0.15
ACC_TOLERANCE_PP = 3.0

# ---------------------------------------------------------------- statistics
N_BOOT = 2000
N_MC = 200
N_MC_CI = 1000
N_REGEN = 10
