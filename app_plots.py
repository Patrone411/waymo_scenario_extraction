import matplotlib.pyplot as plt


def plot_cpna(min_dist, ttc):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(min_dist, bins=10, edgecolor="white", alpha=0.75)
    axes[0].set_title("CPNA: min distance")

    axes[1].hist(ttc, bins=10, range=(0, 10), edgecolor="white", alpha=0.75)
    axes[1].set_title("CPNA: TTC at min distance")

    return fig


def plot_ccrb(ttc, ego_speeds, npc_speeds):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # TTC
    axes[0].hist(ttc, bins=10, range=(0, 10),
                 alpha=0.75, edgecolor="white")
    axes[0].set_title("CCRb: TTC at t₁")

    # Speed
    axes[1].hist(ego_speeds, bins=30, alpha=0.6, label="ego", rwidth=0.9)
    axes[1].hist(npc_speeds, bins=30, alpha=0.6, label="npc", rwidth=0.9)
    axes[1].legend()
    axes[1].set_title("Average speed distribution")

    fig.tight_layout()
    return fig


def plot_cut_in(min_ttc_values, actor_frames):
    af = actor_frames[actor_frames["scenario"] == "cut_in.osc"].copy()

    group_cols = ["scene_id", "segment_id", "t0", "t1"]

    ego_avg_speeds = []
    npc_avg_speeds = []

    for _, group in af.groupby(group_cols):

        ego = group[group["role"] == "ego_vehicle"]
        npc = group[group["role"] == "npc"]

        if not ego.empty:
            ego_avg_speeds.append(ego["speed"].mean() * 3.6)

        if not npc.empty:
            npc_avg_speeds.append(npc["speed"].mean() * 3.6)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(
        min_ttc_values,
        bins=10,
        range=(0, 10),
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].set_xlim(0, 10)
    axes[0].set_xlabel("Minimum TTC after lane change (s)")
    axes[0].set_ylabel("Scenario instances")
    axes[0].set_title("Min TTC after cut-in")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        ego_avg_speeds,
        bins=30,
        alpha=0.65,
        label="ego_vehicle",
        edgecolor="white",
        linewidth=0.5,
    )

    axes[1].hist(
        npc_avg_speeds,
        bins=30,
        alpha=0.65,
        label="npc",
        edgecolor="white",
        linewidth=0.5,
    )

    axes[1].set_xlabel("Average speed (km/h)")
    axes[1].set_ylabel("Scenario instances")
    axes[1].set_title("Average speed distribution")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    return fig

def plot_start_stats(af_filtered):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ─────────────────────────────
    # Speed at t0
    # ─────────────────────────────
    for role in af_filtered["role"].unique():
        spd = af_filtered[af_filtered["role"] == role]["speed"].dropna()
        axes[0].hist(
            spd * 3.6,
            bins=30,
            alpha=0.65,
            label=role,
            edgecolor="white",
            linewidth=0.5,
        )

    axes[0].set_xlabel("Speed at t₀ (km/h)")
    axes[0].set_ylabel("Scenario Instances")
    axes[0].set_title("Speed distribution at t₀")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ─────────────────────────────
    # Scenario duration
    # ─────────────────────────────
    durations = (af_filtered["t1"] - af_filtered["t0"]) / 10.0

    axes[1].hist(
        durations,
        bins=20,
        color="#2196F3",
        edgecolor="white",
        linewidth=0.5,
    )

    axes[1].set_xlabel("Scenario duration (s)")
    axes[1].set_ylabel("Scenario Instances")
    axes[1].set_title("Scenario duration distribution")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    return fig