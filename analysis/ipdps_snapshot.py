"""Generate analysis/generated/results_snapshot.tex from committed decentralized results.

Measured \\R{} values in the IPDPS paper are generated here from committed
result JSONs, including churn, replay, and the closed-form reservation sweep.
"""
from __future__ import annotations

import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "analysis", "generated", "results_snapshot.tex")


def load_fair_compare():
    p = os.path.join(RES, "fair_compare", "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_s1nested():
    p = os.path.join(RES, "s1nested", "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    defs = {}
    ch = load("churn_evaluation.json")
    if ch:
        s = ch["summary"]
        defs["chEvents"] = str(s["committed_topology_events"])
        defs["chEpochs"] = str(s["final_epoch"])
        defs["chSettleRounds"] = str(s["maximum_settling_rounds"])
        defs["chEnergyInc"] = str(s["energy_increase_rounds"])
        defs["chMembers"] = str(s["final_active_members"])
        # strict-neighbor join detail
        for e in ch["events"]:
            if e.get("event") == "join" and "strict_neighbor" in e:
                sn = e["strict_neighbor"]
                defs["chMsgPerRound"] = f"{sn['messages_per_job_round']:.0f}"
                defs["chGapErr"] = f"{sn['final_max_gap_error']:.1e}"
        # per-event settling (strict-neighbor), for the per-event table
        evmap = {"join": "Join", "adjacent-swap": "Swap",
                 "graceful-leave": "Leave", "lease-expiry-repair": "Repair"}
        for e in ch["events"]:
            key = evmap.get(e.get("event"))
            sn = e.get("strict_neighbor")
            if key and sn and sn.get("settling_rounds") is not None:
                defs[f"chEv{key}Rounds"] = str(sn["settling_rounds"])
                defs[f"chEv{key}Ok"] = "yes" if sn.get("settled") else "no"
        wc = ch["work_conservation"]
        defs["chLost"] = f"{wc['lost_work']:.1f}"
        defs["chResidual"] = f"{wc['conservation_residual']:.0e}"
        defs["chFenced"] = str(wc["old_epoch_actions_fenced"])
        defs["chDup"] = str(wc["duplicate_receipts_ignored"])
        defs["chParked"] = f"{wc['durably_parked_backlog']:.2f}"

    sc = load("desync_scale.json")
    if sc:
        pops = {p["n"]: p for p in sc["populations"]}
        for n in (10, 50, 200):
            if n in pops and pops[n]["settling_rounds"] is not None:
                defs[f"scaleR{n}"] = f"{pops[n]['settling_rounds']:,}"
        defs["scaleNmax"] = str(max(pops))
        defs["scaleEinc"] = str(sum(p["energy_increase_rounds"] for p in sc["populations"]))
        defs["scaleErr"] = f"{max(p['final_max_gap_error'] for p in sc['populations']):.0e}"

    ad = load("desync_adversary.json")
    if ad:
        hi = ad["fractions"][-1]                          # highest malicious fraction
        defs["advFrac"] = f"{100*hi['malicious_frac']:.0f}"
        defs["advInflNo"] = f"{hi['no_defense']['attacker_slot_inflation']:.3f}"
        defs["advInflDef"] = f"{hi['defense']['attacker_slot_inflation']:.3f}"
        defs["advJainNo"] = f"{hi['no_defense']['honest_jain']:.3f}"
        defs["advJainDef"] = f"{hi['defense']['honest_jain']:.3f}"
        defs["advMinRatioNo"] = f"{hi['no_defense']['honest_min_ratio']:.3f}"
        defs["advMinRatioDef"] = f"{hi['defense']['honest_min_ratio']:.3f}"

    dd = load("desync_distributed.json")
    if dd:
        import math
        fair = 2 * math.pi / dd["n"]
        defs["distN"] = str(dd["n"])
        defs["distWall"] = f"{dd['wall_seconds']:.0f}"
        defs["distInitErr"] = f"{dd['initial_max_gap_error']:.2f}"
        defs["distFinalPct"] = f"{100*dd['final_max_gap_error']/fair:.1f}"

    as_ = load("desync_async.json")
    if as_:
        sw = as_["sweep"]
        by = {r["loss"]: r for r in sw}
        defs["asyncPct0"] = f"{by[0.0]['final_pct_of_fair']:.1f}"
        hi = max(by)
        defs["asyncMaxLoss"] = f"{100*hi:.0f}"
        defs["asyncPctMax"] = f"{by[hi]['final_pct_of_fair']:.1f}"

    # packing (time-shift mode), if present
    gains = []
    for scale, tag in ((0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0"), (4.0, "4p0")):
        pk = load(f"raw_packing_c{tag}.json")
        if not pk:
            continue
        ps = pk["per_scheme"]
        defs[f"packR{tag}"] = f"{ps['repulsive']['gain_vs_rr_pct']:.1f}"
        gains.append(ps["repulsive"]["gain_vs_rr_pct"])
        defs["packSeeds"] = str(pk["seeds"])
    if gains:
        defs["packMinGain"] = f"{min(gains):.1f}"
        defs["packMaxGain"] = f"{max(gains):.1f}"

    # inference-vs-reservation safety/cost study (phase inference frontier)
    fr = load("phase_frontier.json")
    if fr:
        defs["frPeakNodes"] = f"{fr['peak_nodes_mean']:.0f}"
        defs["frGrid"] = str(len(fr["grid"]))
        defs["frSafe"] = str(fr["safe_points"])
        defs["frDominates"] = "does not" if not fr["dominates_peak"] else "does"
        bs = fr["best_safe_point"]
        if bs:
            defs["frBestNodes"] = f"{bs['nodes_mean']:.0f}"
            defs["frBestDelta"] = f"{abs(bs['nodes_vs_peak_pct']):.1f}"
            defs["frBestSign"] = "more" if bs["nodes_vs_peak_pct"] < 0 else "fewer"
    pf = load("oos_phase_infer.json")
    if pf:
        s = pf["summary"]
        defs["pfGain"] = f"{s['gated_gain_vs_peak_pct']:.0f}"
        defs["pfOver"] = f"{100*s['max_gated_overload']:.0f}"
        adv = pf["adversary"]
        hi = adv[-1]
        defs["pfAdvFrac"] = f"{100*hi['malicious_frac']:.0f}"
        defs["pfAdvOver"] = f"{100*hi['overload']:.0f}"

    abl = load("raw_placement_ablation.json")
    if abl:
        for key, tag in (("lz", "ablLZ"), ("best", "ablBest"), ("first", "ablFirst")):
            row = abl.get("K24_z4", {}).get(key)
            if row:
                defs[tag] = f"{row['nodes']:.1f}"
        try:
            vals = [abl["K24_z4"][k]["nodes"] for k in ("lz", "best", "first")]
            defs["ablSpread"] = f"{100 * (max(vals) - min(vals)) / min(vals):.1f}"
        except Exception:
            pass

    cr = load("coupling_radius.json")
    if cr:
        for pop in cr["populations"]:
            n = pop["n"]
            MAX_R = 4_000_000
            for e in pop["by_k"]:
                k = e["k"]
                converged = e["settling_rounds"] < MAX_R
                defs[f"coupK{k}n{n}r"] = f"{e['settling_rounds']:,}" if converged else "DNF"
                defs[f"coupK{k}n{n}x"] = f"{e['rounds_vs_k1']:.1f}" if converged else "DNF"
        # summary: which k values converge
        defs["coupEvenDNF"] = "k=2,4 do not converge"
        defs["coupOddConverge"] = "k=1,3 converge"

    rp = load("raw_trace_replay.json")
    if rp:
        d = rp["data"]
        defs["rpTasks"] = str(d["tasks"])
        defs["rpIntervals"] = str(d["intervals"])
        defs["rpCommon"] = f"{100*d['common_mode_energy_fraction']:.0f}"
        pol = rp["policies"]
        m = {"rpMMF": "max-min-fair", "rpNbr": "neighbor-token-scaffold",
             "rpPress": "max-pressure"}
        for pre, key in m.items():
            v = pol[key]
            defs[f"{pre}Jain"] = f"{v['fulfillment_jain_index']:.3f}"
            defs[f"{pre}Min"] = f"{v['minimum_task_fulfillment']:.3f}"
            defs[f"{pre}Msgs"] = str(v["directed_neighbor_messages"])

    # Closed-form reservation comparison. The last_day=8 files repeat held day
    # 6 and are explicitly excluded by the committed summary's provenance.
    sr = load("s1robust/stage1_robustness_summary.json")
    if sr:
        rows = [r for r in sr["runs"] if r["last_day"] < 8]
        assert len(rows) == sr["distinct_configs"] == 12
        defs["cfConfigs"] = str(len(rows))
        first_run = load("s1robust/" + rows[0]["file"])
        defs["cfTasks"] = str(first_run["meta"]["n_tasks"])
        defs["cfDays"] = str(len({r["held_day"] for r in rows}))
        defs["cfSeeds"] = str(len({r["seed"] for r in rows}))
        defs["cfWindowsMin"] = f"{min(r['test_windows'] for r in rows):,}".replace(",", "{,}")
        defs["cfWindowsMax"] = f"{max(r['test_windows'] for r in rows):,}".replace(",", "{,}")
        defs["cfPeakMean"] = f"{statistics.mean(r['peak']['nodes'] for r in rows):.1f}"
        defs["cfHarmMean"] = f"{statistics.mean(r['harmonic_z2']['nodes'] for r in rows):.1f}"
        defs["cfRed"] = f"{sr['harmonic_z2_vs_peak_node_reduction_pct']['mean']:.1f}"
        defs["cfRedMin"] = f"{sr['harmonic_z2_vs_peak_node_reduction_pct']['min']:.1f}"
        defs["cfRedMax"] = f"{sr['harmonic_z2_vs_peak_node_reduction_pct']['max']:.1f}"
        defs["cfZeroOver"] = str(sum(r["harmonic_z2"]["overload"] == 0 for r in rows))
        defs["cfHarmBeatsMean"] = str(sum(r["harmonic_z2"]["nodes"] < r["mean_z2"]["nodes"] for r in rows))
        nonzero = [r for r in rows if r["harmonic_z2"]["overload"] > 0]
        assert len(nonzero) == 1
        defs["cfHarmOverOne"] = f"{100 * nonzero[0]['harmonic_z2']['overload']:.4f}"
        defs["cfPeakOverOne"] = f"{100 * nonzero[0]['peak']['overload']:.4f}"

    # Rhythm prevalence behind the workload-analysis section and the
    # phase-velocity caption. The 8,225 count was previously typed into the
    # caption with no committed source; results/rhythmic_task_count.json now
    # reproduces the RHYTHMIC selector used for the phase-velocity figure.
    rt = load("rhythmic_task_count.json")
    if rt:
        def thou(n):
            return format(n, ",").replace(",", "{,}")
        frac = rt["rhythmic_fraction_of_filtered"]
        med = rt["median_share_rhythmic"]
        defs["rhyTasks"] = thou(rt["rhythmic_tasks"])
        defs["rhyFiltered"] = thou(rt["tasks_after_activity_filter"])
        defs["rhyTotal"] = thou(rt["tasks_in_file"])
        defs["rhyPct"] = format(100 * frac, ".1f")
        defs["rhyShareThresh"] = str(rt["share_threshold"])
        defs["rhyShareMed"] = format(med, ".2f")
    # Nested selection of the reservation margin and harmonic order: both
    # chosen on the validation day only, then scored once on the held-out
    # day. Replaces a figure whose margin had been selected on the held-out
    # days themselves.
    ns = load_s1nested()
    if ns:
        r = ns["reduction_pct"]
        fx = ns["paper_fixed_z2_K4"]
        k4 = ns["fixed_K4_val_selected_z"]
        defs["nsConfigs"] = str(ns["n_evaluated"])
        defs["nsInfeasible"] = str(ns["n_infeasible"])
        defs["nsDays"] = str(len(ns["reduction_by_held_day"]))
        defs["nsRed"] = format(r["mean"], ".1f")
        defs["nsRedSd"] = format(r["sd"], ".1f")
        defs["nsRedMin"] = format(r["min"], ".1f")
        defs["nsRedMax"] = format(r["max"], ".1f")
        defs["nsZeroOver"] = str(ns["zero_overload_heldout"])
        defs["nsPeakZeroOver"] = str(ns["peak_zero_overload_heldout"])
        defs["nsFixedRed"] = format(fx["mean"], ".1f")
        defs["nsFixedSd"] = format(fx["sd"], ".1f")
        defs["nsK4Red"] = format(k4["mean"], ".1f")
        zc = ns["chosen_z_counts"]
        kc = ns["chosen_K_counts"]
        zmode = max(zc, key=lambda k: zc[k])
        kmode = max(kc, key=lambda k: kc[k])
        defs["nsZMode"] = str(zmode)
        defs["nsZModeN"] = str(zc[zmode])
        defs["nsKMode"] = str(kmode)
        defs["nsKModeN"] = str(kc[kmode])
        pk = ns["per_K_val_selected_z"]
        vals = [v["mean_reduction"] for v in pk.values()]
        defs["nsKSpread"] = format(max(vals) - min(vals), ".1f")
    # The packing number comes from a centralized chooser over a fixed shift
    # grid, not from the ring. Emit the grid size and the even-splay arm so the
    # paper can attribute the gain to the code that produced it.
    pk1 = load("raw_packing_c1p0.json")
    if pk1:
        defs["packGrid"] = "48"
        leg = pk1["per_scheme"]["legacy_desync"]
        defs["packLegacy"] = format(leg["gain_vs_rr_pct"], ".2f")
        ci = leg.get("gain_ci95")
        if ci:
            defs["packLegacyCIlo"] = format(ci[0], ".2f")
            defs["packLegacyCIhi"] = format(ci[1], ".2f")
    pv = load("phase_velocity_plot.json")
    if pv:
        defs["pvTasks"] = str(pv["sampled_tasks"])
        defs["pvPairs"] = format(pv["window_pairs"], ",")
        defs["pvHours"] = format(pv["window_hours"], ".0f")
    fc = load_fair_compare()
    if fc:
        ag = fc["aggregate_4arm_configs"]["arms"]
        defs["fcConfigs"] = str(fc["aggregate_4arm_configs"]["n_configs"])
        for key, arm in (("Peak", "peak"), ("Mean", "mean_z"),
                         ("Harm", "harmonic_z"), ("Neur", "neural")):
            a = ag[arm]
            defs["fc" + key + "Nodes"] = format(a["mean_nodes"], ".1f")
            defs["fc" + key + "Zero"] = str(a["n_zero_overload"])
            if arm != "peak":
                defs["fc" + key + "Pct"] = format(abs(a["mean_pct_vs_peak"]), ".1f")
        defs["fcNeurWins"] = str(ag["neural"]["wins"])
        defs["fcHarmWins"] = str(ag["harmonic_z"]["wins"])
        defs["fcMeanWins"] = str(ag["mean_z"]["wins"])
        defs["fcPeakWins"] = str(ag["peak"]["wins"])
        defs["fcNeurSafeBeatPeak"] = str(sum(
            c["arms"]["neural"]["nodes"] < c["arms"]["peak"]["nodes"]
            and c["arms"]["neural"]["overload"] <= c["arms"]["peak"]["overload"]
            for c in fc["configs"] if c["status"] == "ok" and "neural" in c["arms"]
        ))
    # A completed 16-worker CloudLab UDP-ring *component* campaign. These
    # are not scheduling, job-latency, or cross-node-message measurements.
    cl = load("cloudlab16oct09/kadence16oct09-results-v2/summary.json")
    if cl:
        assert cl["complete"] and cl["valid_cells"] == cl["expected_cells"]
        defs["cl16Cells"] = str(cl["valid_cells"])
        cm = load("cloudlab16oct09/kadence16oct09-results-v2/manifest.json")
        assert cm and len(cm["workers"]) == 16
        defs["cl16Workers"] = str(len(cm["workers"]))
        defs["cl16Reps"] = str(cm["repetitions_per_condition"])
        defs["cl16PerCondition"] = str(len(cm["workers"]) * cm["repetitions_per_condition"])
        defs["cl16WindowSec"] = format(cm["seconds_per_repetition"], ".0f")
        defs["cl16JitterMs"] = format(cm["jitter"] * 1000, ".0f")
        for n in cm["jobs_per_ring"]:
            defs[f"cl16N{n}"] = str(n)
        defs["cl16Loss0Pct"] = format(100 * min(cm["loss_levels"]), ".0f")
        defs["cl16Loss30Pct"] = format(100 * max(cm["loss_levels"]), ".0f")
        by_condition = {(c["jobs_per_ring"], c["loss"]): c
                        for c in cl["conditions"]}
        for jobs, loss, tag in ((8, 0.0, "J8L0"), (16, 0.0, "J16L0"),
                                (48, 0.0, "J48L0"), (48, 0.3, "J48L30")):
            c = by_condition[(jobs, loss)]
            assert c["complete"] and c["n_valid"] == 5
            defs[f"cl16{tag}Med"] = format(c["mean_median_error_pct_of_fair"], ".1f")
            # The classifier checks sorted-gap convergence before identity
            # order, so this is a LOWER BOUND on broken-order rings.
            defs[f"cl16{tag}BrokenMin"] = str(c["classification_counts"].get("order-broken", 0))
    # Admission-kernel timing on the local CPU, not an end-to-end scheduler.
    nb = load("../src/native/results/native_bench.json")
    if nb:
        for arm, tag in (("reserve_peak", "Peak"),
                         ("reserve_slot", "Slot"),
                         ("reserve_harmonic_k4", "Harm")):
            for lang, pre in (("c", "nc"), ("python", "np")):
                rows = [r for r in nb["runs"] if r["impl"] == arm
                        and r["lang"] == lang and r["batch"] == 1]
                assert len(rows) == 1
                defs[f"{pre}{tag}P50Us"] = format(rows[0]["p50_ns"] / 1000, ".2f")
                defs[f"{pre}{tag}P99Us"] = format(rows[0]["p99_ns"] / 1000, ".2f")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("% Auto-generated by analysis/ipdps_snapshot.py. Do not edit by hand.\n")
        for k, v in sorted(defs.items()):
            f.write(f"\\defR{{{k}}}{{{v}}}\n")
    print(f"wrote {len(defs)} macros to {OUT}")


if __name__ == "__main__":
    main()
