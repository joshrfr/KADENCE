/// KADENCE: decentralised desynchronisation ring scheduler, Rust prototype.
///
/// Implements the same strict-neighbour phase-oscillator algorithm as the
/// Python kernel in src/kadence/neighbor_gossip.py, using async Tokio tasks instead
/// of OS processes.  One Tokio task per logical job.
///
/// Update rule (paper convention): displacement = (alpha / 2) * correction,
/// clipped to beta = 0.45 of the free slack on the side being moved toward.
/// alpha = 1 is the paper's configuration; the pre-fix binary hardcoded a
/// multiplier of 0.1, which is alpha = 0.2 in this convention.
///
/// Build:  cargo build --release
/// Run:    kadence --jobs 48 --seconds 10 --reps 1 --alpha 1 --loss 0.1
///         kadence --jobs 16 --phases-file init.txt --trace-every 5
///
/// Every agent counts the datagrams it actually sends and receives and the
/// ticks it actually executes; the ring result aggregates them.
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use tokio::time;

const TWO_PI: f64 = std::f64::consts::TAU;
const SAFETY_FRACTION: f64 = 0.45;
const DEFAULT_BASE_PORT: u16 = 52000;
/// Strict criterion used by the simulation (absolute max gap error, radians).
const CRIT_STRICT_ABS: f64 = 1e-6;
/// Loose criterion used by the CloudLab runtime: err < 0.1 * (2*pi/n).
const CRIT_LOOSE_FRAC: f64 = 0.1;

// ---------------------------------------------------------------------------
// Kernel math (mirrors src/kadence/neighbor_gossip.py exactly)
// ---------------------------------------------------------------------------

#[inline]
fn forward_gap(left: f64, right: f64) -> f64 {
    (right - left).rem_euclid(TWO_PI)
}

#[inline]
fn local_correction(left: f64, cur: f64, right: f64, target: f64) -> f64 {
    let left_err = forward_gap(left, cur) - target;
    let right_err = forward_gap(cur, right) - target;
    right_err - left_err
}

#[inline]
fn limit_displacement(requested: f64, left: f64, cur: f64, right: f64) -> f64 {
    if requested >= 0.0 {
        let available = f64::max(0.0, forward_gap(cur, right));
        f64::min(requested, SAFETY_FRACTION * available)
    } else {
        let available = f64::max(0.0, forward_gap(left, cur));
        f64::max(requested, -SAFETY_FRACTION * available)
    }
}

fn gap_error(phases: &[f64], target: f64) -> f64 {
    let mut s: Vec<f64> = phases.iter().map(|&p| p.rem_euclid(TWO_PI)).collect();
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let mut max_err: f64 = 0.0;
    for i in 0..n {
        let gap = if i + 1 < n { s[i + 1] - s[i] } else { s[0] + TWO_PI - s[n - 1] };
        max_err = f64::max(max_err, (gap - target).abs());
    }
    max_err
}

/// True iff the cyclic order of `phases` (indexed by job) is still the
/// initial order 0,1,..,n-1 (a rotation of the sorted order is allowed).
fn order_preserved(phases: &[f64]) -> bool {
    let n = phases.len();
    let p: Vec<f64> = phases.iter().map(|&x| x.rem_euclid(TWO_PI)).collect();
    // number of descents around the cycle must be exactly 1 (or 0 if all equal)
    let mut descents = 0;
    for i in 0..n {
        if p[(i + 1) % n] < p[i] {
            descents += 1;
        }
    }
    descents <= 1
}

// ---------------------------------------------------------------------------
// Per-job async agent
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Cfg {
    n: usize,
    base_port: u16,
    target: f64,
    seconds: f64,
    tick_ms: u64,
    jitter_ms: f64,
    loss: f64,
    alpha: f64,
    seed: u64,
}

#[derive(Default)]
struct AgentStats {
    phase: f64,
    ticks: u64,
    sent: u64,
    received: u64,
    malformed: u64,
    loss_dropped: u64,
    send_errors: u64,
    residual_at_exit: u64,
    recv_nonloopback_src: u64,
    sent_nonloopback_dst: u64,
    trace: Vec<f64>,
}

/// Parse one datagram and apply it. Returns false if it was malformed.
fn apply_datagram(bytes: &[u8], left_phase: &mut f64, right_phase: &mut f64) -> bool {
    let Ok(s) = std::str::from_utf8(bytes) else { return false };
    let mut parts = s.splitn(2, ':');
    let (Some(who), Some(v)) = (parts.next(), parts.next()) else { return false };
    let Ok(val) = v.parse::<f64>() else { return false };
    match who {
        "L" => *left_phase = val,
        "R" => *right_phase = val,
        _ => return false,
    }
    true
}

async fn agent(idx: usize, phase0: f64, cfg: Cfg, sock: UdpSocket) -> AgentStats {
    let n = cfg.n;
    let left_addr: SocketAddr =
        format!("127.0.0.1:{}", cfg.base_port + ((idx + n - 1) % n) as u16).parse().unwrap();
    let right_addr: SocketAddr =
        format!("127.0.0.1:{}", cfg.base_port + ((idx + 1) % n) as u16).parse().unwrap();

    let mut st = AgentStats::default();
    let mut phase = phase0;
    let mut left_phase = (phase0 - cfg.target).rem_euclid(TWO_PI);
    let mut right_phase = (phase0 + cfg.target).rem_euclid(TWO_PI);
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ (idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let mut buf = [0u8; 64];

    let tick = Duration::from_millis(cfg.tick_ms);
    let deadline = Instant::now() + Duration::from_secs_f64(cfg.seconds);
    let mut next_tick = Instant::now();

    loop {
        time::sleep_until(time::Instant::from_std(next_tick)).await;
        if Instant::now() >= deadline {
            break;
        }

        // FIX 1: drain the socket so that EVERY datagram is parsed. The old
        // code read a second datagram only to test for emptiness and threw it
        // away, so every second message was lost.
        loop {
            match sock.recv_from(&mut buf) {
                Ok((len, src)) => {
                    st.received += 1;
                    if !src.ip().is_loopback() {
                        st.recv_nonloopback_src += 1;
                    }
                    if !apply_datagram(&buf[..len], &mut left_phase, &mut right_phase) {
                        st.malformed += 1;
                    }
                }
                Err(_) => break, // WouldBlock: nothing left
            }
        }

        // Send to neighbours (with optional injected loss).
        if rng.gen::<f64>() >= cfg.loss {
            match sock.send_to(format!("R:{phase}").as_bytes(), left_addr) {
                Ok(_) => {
                    st.sent += 1;
                    if !left_addr.ip().is_loopback() {
                        st.sent_nonloopback_dst += 1;
                    }
                }
                Err(_) => st.send_errors += 1,
            }
        } else {
            st.loss_dropped += 1;
        }
        if rng.gen::<f64>() >= cfg.loss {
            match sock.send_to(format!("L:{phase}").as_bytes(), right_addr) {
                Ok(_) => {
                    st.sent += 1;
                    if !right_addr.ip().is_loopback() {
                        st.sent_nonloopback_dst += 1;
                    }
                }
                Err(_) => st.send_errors += 1,
            }
        } else {
            st.loss_dropped += 1;
        }

        // Kernel update. FIX 2: rate is a parameter, paper convention alpha/2.
        let corr = local_correction(left_phase, phase, right_phase, cfg.target);
        let disp = limit_displacement(0.5 * cfg.alpha * corr, left_phase, phase, right_phase);
        phase = (phase + disp).rem_euclid(TWO_PI);
        st.ticks += 1;
        st.trace.push(phase);

        let jit = if cfg.jitter_ms > 0.0 { rng.gen::<f64>() * cfg.jitter_ms / 1000.0 } else { 0.0 };
        next_tick = Instant::now() + tick + Duration::from_secs_f64(jit);
    }

    // Count what is still queued at exit (sent but never applied).
    while sock.recv_from(&mut buf).is_ok() {
        st.residual_at_exit += 1;
    }
    st.phase = phase;
    st
}

// ---------------------------------------------------------------------------
// One-ring runner
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct RingResult {
    init_gap_error: f64,
    final_gap_error: f64,
    final_pct_of_fair: f64,
    target: f64,
    converged_loose_0p1_target: bool,
    converged_strict_1e6: bool,
    order_preserved: bool,
    wall_s: f64,
    ticks_executed: u64,
    ticks_expected_per_job: u64,
    ticks_min_per_job: u64,
    ticks_max_per_job: u64,
    tick_deficit_total: i64,
    tick_deficit_frac: f64,
    datagrams_sent: u64,
    datagrams_received: u64,
    datagrams_malformed: u64,
    sends_dropped_by_loss_injection: u64,
    send_errors: u64,
    unread_at_exit: u64,
    cross_node_recv_nonloopback_src: u64,
    cross_node_sent_nonloopback_dst: u64,
    sent_not_received_or_unread: i64,
    recv_min_per_job: u64,
    tick_of_first_loose: Option<usize>,
    tick_of_first_strict: Option<usize>,
    err_trace: Option<Vec<(usize, f64)>>,
    ticks_per_job: Vec<u64>,
    recv_per_job: Vec<u64>,
    final_phases: Vec<f64>,
}

fn load_phases(path: &str) -> Vec<f64> {
    std::fs::read_to_string(path)
        .expect("cannot read --phases-file")
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("bad float in --phases-file"))
        .collect()
}

async fn run_ring(cfg: &Cfg, phases_file: &Option<String>, trace_every: usize) -> RingResult {
    let n = cfg.n;
    let mut phases: Vec<f64> = match phases_file {
        Some(p) => {
            let v = load_phases(p);
            assert_eq!(v.len(), n, "--phases-file has {} values, need --jobs {}", v.len(), n);
            v
        }
        None => {
            let mut rng = StdRng::seed_from_u64(cfg.seed);
            (0..n).map(|_| rng.gen::<f64>() * TWO_PI).collect()
        }
    };
    phases.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let init_err = gap_error(&phases, cfg.target);

    // Bind every socket up front so no agent sends before its neighbour exists.
    let socks: Vec<UdpSocket> = (0..n)
        .map(|i| {
            let s = UdpSocket::bind(format!("127.0.0.1:{}", cfg.base_port + i as u16))
                .unwrap_or_else(|e| panic!("bind port {}: {e}", cfg.base_port + i as u16));
            s.set_nonblocking(true).unwrap();
            s
        })
        .collect();

    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(n);
    for (i, sock) in socks.into_iter().enumerate() {
        handles.push(tokio::spawn(agent(i, phases[i], cfg.clone(), sock)));
    }
    let mut stats: Vec<AgentStats> = Vec::with_capacity(n);
    for h in handles {
        stats.push(h.await.unwrap());
    }
    let wall = t0.elapsed().as_secs_f64();

    let final_phases: Vec<f64> = stats.iter().map(|s| s.phase).collect();
    let final_err = gap_error(&final_phases, cfg.target);

    let ticks: Vec<u64> = stats.iter().map(|s| s.ticks).collect();
    let ticks_executed: u64 = ticks.iter().sum();
    // Budget: one tick at t=0 and one every tick_ms while t < seconds.
    let expected = ((cfg.seconds * 1000.0) / cfg.tick_ms as f64).ceil() as u64;
    let deficit = (expected * n as u64) as i64 - ticks_executed as i64;

    let sent: u64 = stats.iter().map(|s| s.sent).sum();
    let recv: u64 = stats.iter().map(|s| s.received).sum();
    let resid: u64 = stats.iter().map(|s| s.residual_at_exit).sum();

    // Ring-level error trace (agents are asynchronous; tick k means each
    // agent's k-th update, a standard async-round alignment).
    let min_t = *ticks.iter().min().unwrap() as usize;
    let mut first_loose = None;
    let mut first_strict = None;
    let mut trace_out = if trace_every > 0 { Some(Vec::new()) } else { None };
    for k in 0..min_t {
        let col: Vec<f64> = stats.iter().map(|s| s.trace[k]).collect();
        let e = gap_error(&col, cfg.target);
        if first_loose.is_none() && e < CRIT_LOOSE_FRAC * cfg.target {
            first_loose = Some(k);
        }
        if first_strict.is_none() && e < CRIT_STRICT_ABS {
            first_strict = Some(k);
        }
        if let Some(t) = trace_out.as_mut() {
            if k % trace_every == 0 || k + 1 == min_t {
                t.push((k, e));
            }
        }
    }

    RingResult {
        init_gap_error: init_err,
        final_gap_error: final_err,
        final_pct_of_fair: final_err / cfg.target * 100.0,
        target: cfg.target,
        converged_loose_0p1_target: final_err < CRIT_LOOSE_FRAC * cfg.target,
        converged_strict_1e6: final_err < CRIT_STRICT_ABS,
        order_preserved: order_preserved(&final_phases),
        wall_s: wall,
        ticks_executed,
        ticks_expected_per_job: expected,
        ticks_min_per_job: *ticks.iter().min().unwrap(),
        ticks_max_per_job: *ticks.iter().max().unwrap(),
        tick_deficit_total: deficit,
        tick_deficit_frac: deficit as f64 / (expected * n as u64) as f64,
        datagrams_sent: sent,
        datagrams_received: recv,
        datagrams_malformed: stats.iter().map(|s| s.malformed).sum(),
        sends_dropped_by_loss_injection: stats.iter().map(|s| s.loss_dropped).sum(),
        send_errors: stats.iter().map(|s| s.send_errors).sum(),
        unread_at_exit: resid,
        cross_node_recv_nonloopback_src: stats.iter().map(|s| s.recv_nonloopback_src).sum(),
        cross_node_sent_nonloopback_dst: stats.iter().map(|s| s.sent_nonloopback_dst).sum(),
        sent_not_received_or_unread: sent as i64 - recv as i64 - resid as i64,
        recv_min_per_job: stats.iter().map(|s| s.received).min().unwrap(),
        tick_of_first_loose: first_loose,
        tick_of_first_strict: first_strict,
        err_trace: trace_out,
        ticks_per_job: ticks.clone(),
        recv_per_job: stats.iter().map(|s| s.received).collect(),
        final_phases,
    }
}

// ---------------------------------------------------------------------------
// CLI + main
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(about = "KADENCE Rust prototype, phase-oscillator ring scheduler")]
struct Args {
    /// Jobs per ring
    #[arg(long, default_value_t = 48)]
    jobs: usize,

    /// Recorded in provenance only; each rep runs one ring (the old flag was
    /// likewise unused).
    #[arg(long, default_value_t = 1)]
    rings: usize,

    /// Wall-clock seconds per run
    #[arg(long, default_value_t = 10.0)]
    seconds: f64,

    /// Tick interval in milliseconds
    #[arg(long, default_value_t = 20)]
    tick_ms: u64,

    /// Extra uniform random delay [0, jitter_ms] added to each tick
    #[arg(long, default_value_t = 0.0)]
    jitter_ms: f64,

    /// Message loss probability [0,1)
    #[arg(long, default_value_t = 0.0)]
    loss: f64,

    /// Update rate alpha, paper convention: displacement = alpha/2 * correction.
    /// Default 1.0 = paper configuration. (Pre-fix binary: fixed 0.1*corr = alpha 0.2.)
    #[arg(long, default_value_t = 1.0)]
    alpha: f64,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Run repetitions
    #[arg(long, default_value_t = 5)]
    reps: usize,

    /// First UDP port (jobs use base_port .. base_port+jobs-1)
    #[arg(long, default_value_t = DEFAULT_BASE_PORT)]
    base_port: u16,

    /// File of whitespace-separated initial phases (length = jobs). Lets
    /// another implementation start from the identical initial condition.
    #[arg(long)]
    phases_file: Option<String>,

    /// Emit the ring error every K ticks in the JSON (0 = off)
    #[arg(long, default_value_t = 0)]
    trace_every: usize,

    /// Write JSON results to this file (- = stdout)
    #[arg(long, default_value = "-")]
    out: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let target = TWO_PI / args.jobs as f64;
    let mut reps = Vec::new();
    for r in 0..args.reps {
        let cfg = Cfg {
            n: args.jobs,
            base_port: args.base_port,
            target,
            seconds: args.seconds,
            tick_ms: args.tick_ms,
            jitter_ms: args.jitter_ms,
            loss: args.loss,
            alpha: args.alpha,
            seed: args.seed + r as u64,
        };
        let res = run_ring(&cfg, &args.phases_file, args.trace_every).await;
        eprintln!(
            "rep {r}: init={:.3}% final={:.3}% loose={} strict={} ticks={}/{} sent={} recv={} wall={:.1}s",
            res.init_gap_error / target * 100.0,
            res.final_pct_of_fair,
            res.converged_loose_0p1_target,
            res.converged_strict_1e6,
            res.ticks_executed,
            res.ticks_expected_per_job * args.jobs as u64,
            res.datagrams_sent,
            res.datagrams_received,
            res.wall_s
        );
        reps.push(res);
    }

    let n = reps.len() as f64;
    let result = serde_json::json!({
        "provenance": {
            "kind": "KADENCE Rust prototype (tokio async UDP)",
            "kernel": "core::neighbor_gossip (local_correction + limit_displacement)",
            "jobs": args.jobs, "rings": args.rings, "seconds": args.seconds,
            "tick_ms": args.tick_ms, "jitter_ms": args.jitter_ms, "loss": args.loss,
            "alpha": args.alpha, "alpha_convention": "displacement = alpha/2 * correction",
            "safety_fraction": SAFETY_FRACTION, "seed": args.seed, "reps": args.reps,
            "criteria": {
                "loose": "final_gap_error < 0.1 * (2*pi/n)  [CloudLab runtime]",
                "strict": "final_gap_error < 1e-6 radians    [simulation]"
            },
            "global_clock": false, "language": "Rust/tokio",
            "version": "fixed: all datagrams parsed, alpha is a parameter"
        },
        "reps": reps,
        "mean_final_pct": reps.iter().map(|r| r.final_pct_of_fair).sum::<f64>() / n,
        "converged_fraction_loose": reps.iter().filter(|r| r.converged_loose_0p1_target).count() as f64 / n,
        "converged_fraction_strict": reps.iter().filter(|r| r.converged_strict_1e6).count() as f64 / n,
    });
    let json = serde_json::to_string_pretty(&result).unwrap();
    if args.out == "-" {
        println!("{json}");
    } else {
        std::fs::write(&args.out, &json).unwrap();
        eprintln!("wrote {}", args.out);
    }
}
