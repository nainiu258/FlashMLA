#!/usr/bin/env python3
"""Convert the SM90 sparse-prefill device trace into a Perfetto trace.

The kernel (built with ``FLASH_MLA_TRACE_PERFETTO=1``) writes 24-byte
``TraceEvent`` records to per-warpgroup regions of a global buffer. This module re-encodes those
records as a self-contained Perfetto trace (protobuf ``Trace``) that can be
opened directly at https://ui.perfetto.dev.

There are no third-party dependencies: the Perfetto protobuf is written with a
tiny hand-rolled encoder for exactly the handful of messages/fields we need.

Typical usage (see tests/profile_sm90_sparse_prefill_perfetto.py for the full
pipeline that runs the kernel):

    from tools.emit_perfetto_trace import build_perfetto_trace
    trace_bytes = build_perfetto_trace(records)   # records: np structured array
    open("out.perfetto-trace", "wb").write(trace_bytes)

The 24-byte record layout must stay in sync with ``TraceEvent`` in
``csrc/sm90/prefill/sparse/phase1.cuh``.
"""

import argparse
import sys

import numpy as np

# Must match `struct TraceEvent` in csrc/sm90/prefill/sparse/phase1.cuh.
#   uint64 ts_ns; uint32 event_id; uint32 cta_id; uint32 role; uint32 iter;
EVENT_DTYPE = np.dtype(
    [
        ("ts", "<u8"),
        ("event", "<u4"),
        ("cta", "<u4"),
        ("role", "<u4"),
        ("iter", "<u4"),
    ]
)
assert EVENT_DTYPE.itemsize == 24, EVENT_DTYPE.itemsize

# Must match `enum TraceEventId` in phase1.cuh.
EVENT_NAMES = {
    0: "Q_TMA_ISSUE",
    1: "Q_TMA_DONE",
    2: "WAIT_FREE_V0L (feeds WG0)",
    3: "COPY_ISSUE_V0L (feeds WG0)",
    4: "WAIT_FREE_V1R (feeds WG1)",
    5: "COPY_ISSUE_V1R (feeds WG1)",
    6: "WAIT_FREE_V0R (feeds WG0)",
    7: "COPY_ISSUE_V0R (feeds WG0)",
    8: "WAIT_FREE_V1L (feeds WG1)",
    9: "COPY_ISSUE_V1L (feeds WG1)",
    10: "QKT_DONE",
    11: "SOFTMAX_DONE",
    12: "PV_LOCAL_ISSUED",
    13: "PV_REMOTE_ISSUED",
    14: "REDUCE_L_DONE",
    15: "STORE_O_DONE",
    16: "WAIT_K_BEGIN",
    17: "WAIT_K_END",
    18: "WAIT_SM_BEGIN",
    19: "WAIT_SM_END",
    20: "WAIT_SS_BEGIN",
    21: "WAIT_SS_END",
    22: "WAIT_FREE_V0L_END (feeds WG0)",
    23: "WAIT_FREE_V1R_END (feeds WG1)",
    24: "WAIT_FREE_V0R_END (feeds WG0)",
    25: "WAIT_FREE_V1L_END (feeds WG1)",
    26: "SIGNAL_K0_FREE_L",
    27: "SIGNAL_K1_FREE_L",
    28: "SIGNAL_K1_FREE_R",
    29: "SIGNAL_K0_FREE_R",
    30: "SIGNAL_WG0_SM",
    31: "SIGNAL_WG0_SS",
    32: "SIGNAL_WG1_SM",
    33: "SIGNAL_WG1_SS",
    34: "SIGNAL_K0_READY_L (feeds WG0)",
    35: "SIGNAL_K1_READY_R (feeds WG1)",
    36: "SIGNAL_K0_READY_R (feeds WG0)",
    37: "SIGNAL_K1_READY_L (feeds WG1)",
    38: "LSE_TMA_STORE_ISSUED",
    39: "PV_LOCAL_DONE",
    40: "PV_REMOTE_DONE",
}

# Explicit bubble (stall) phases: (begin_event, end_event) -> slice name.  These
# BEGIN/END markers are emitted by the kernel AROUND each blocking wait, so their
# duration is real stall time (not an inferred gap between unrelated milestones).
BUBBLE_PAIRS = {
    (0, 1): "load Q",          # Q_TMA_ISSUE -> Q_TMA_DONE
    (2, 22): "wait V0L free",  # precisely around bar_k0_free[0].wait()
    (4, 23): "wait V1R free",
    (6, 24): "wait V0R free",
    (8, 25): "wait V1L free",
    (16, 17): "wait K",        # WAIT_K_BEGIN -> WAIT_K_END
    (18, 19): "wait sM",       # WAIT_SM_BEGIN -> WAIT_SM_END
    (20, 21): "wait sS",       # WAIT_SS_BEGIN -> WAIT_SS_END
}
_END_TO_BEGIN = {end: begin for (begin, end) in BUBBLE_PAIRS}

# Work phases: the marker ends the active interval immediately preceding it.
# The producer's COPY_ISSUE markers end the four cp.async issue regions, so its
# track does not collapse into one undifferentiated "active" bar.
# This does not claim that WGMMA issue/completion is synchronous; it only names
# the warpgroup's observed control-flow interval.
WORK_PHASES = {
    3: "cp.async V0L",
    5: "cp.async V1R",
    7: "cp.async V0R",
    9: "cp.async V1L",
    10: "QK^T GEMM",   # QKT_DONE
    11: "softmax",     # SOFTMAX_DONE
    12: "PV local issue",
    13: "PV remote issue",
    14: "reduce L",    # REDUCE_L_DONE
    15: "store O",     # STORE_O_DONE
    38: "LSE/max TMA store",
}

ROLE_NAMES = {0: "producer", 1: "wg0", 2: "wg1"}

SIGNAL_EVENTS = frozenset(range(26, 38))
COMPLETION_EVENTS = frozenset((39, 40))

# Names for active control-flow regions that do not end in one of the explicit
# WORK_PHASES markers. Keys are (event ending the preceding wait, event
# beginning the next wait). These are common to WG0/WG1; the exact buffer side
# is still available from the instant-mode markers.
ACTIVE_TRANSITIONS = {
    (1, 16): "setup after Q load",
    (17, 16): "QK^T issue",
    (10, 18): "mask scores",
    (12, 18): "await PV local / free buffer",
    (12, 20): "publish scores",
    (19, 20): "rescale + publish scores",
    (13, 16): "await PV remote / free buffer",
    (3, 4): "post-copy bookkeeping",
    (5, 6): "post-copy bookkeeping",
    (7, 8): "post-copy bookkeeping",
    (9, 2): "post-copy bookkeeping",
}

# Perfetto TrackEvent.Type values.
TYPE_SLICE_BEGIN = 1
TYPE_SLICE_END = 2
TYPE_INSTANT = 3


# --------------------------------------------------------------------------- #
# Minimal protobuf encoder (wire types 0 = varint, 2 = length-delimited).    #
# --------------------------------------------------------------------------- #
def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _u32(field: int, v: int) -> bytes:
    return _varint((field << 3) | 0) + _varint(v)


def _u64(field: int, v: int) -> bytes:
    return _varint((field << 3) | 0) + _varint(v)


def _enum(field: int, v: int) -> bytes:
    return _varint((field << 3) | 0) + _varint(v)


def _string(field: int, s: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(s)) + s


def _msg(field: int, m: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(m)) + m


# Every packet belongs to a "trusted packet sequence"; Perfetto drops packets
# without a trusted_packet_sequence_id. A single synthetic producer (id 1) is
# enough for a whole hand-written trace.
SEQ_ID = 1


def _thread_descriptor(pid: int, tid: int, name: str) -> bytes:
    # ThreadDescriptor.pid=1, .tid=2, .thread_name=5.
    b = _u32(1, pid)
    b += _u32(2, tid)
    b += _string(5, name.encode())
    return b


def _track_descriptor_packet(uuid: int, name: str) -> bytes:
    # TrackDescriptor.uuid=1, .name=2, .thread=4. Do NOT use field 3 (that is
    # `process`, a ProcessDescriptor that requires a pid) or field 5 (parent_uuid).
    td = _u64(1, uuid)
    td += _string(2, name.encode())
    td += _msg(4, _thread_descriptor(1, uuid, name))
    # TracePacket.trusted_packet_sequence_id=10, .track_descriptor=60.
    pkt = _u32(10, SEQ_ID)
    pkt += _msg(60, td)
    return pkt


def _track_event_packet(ts_ns: int, track_uuid: int, type_: int, name: str) -> bytes:
    # TrackEvent.track_uuid=11, .type=9, .name=23 (legacy string).
    te = _u64(11, track_uuid)
    te += _enum(9, type_)
    te += _string(23, name.encode())
    # TracePacket.trusted_packet_sequence_id=10, .timestamp=8 (ns), .track_event=11.
    pkt = _u32(10, SEQ_ID)
    pkt += _u64(8, ts_ns)
    pkt += _msg(11, te)
    return pkt


def _wrap_trace(packets) -> bytes:
    out = b""
    for p in packets:
        out += _msg(1, p)  # Trace.packet = 1
    return out


# --------------------------------------------------------------------------- #
# Trace building.                                                            #
# --------------------------------------------------------------------------- #
def _event_label(event_id: int, it: int) -> str:
    base = EVENT_NAMES.get(int(event_id), f"EVENT_{event_id}")
    return f"{base}#{it}"


def build_perfetto_trace(records, mode: str = "instant") -> bytes:
    """records: numpy structured array with EVENT_DTYPE (or a raw (N, 5) array
    ordered as [ts, event, cta, role, iter]). Returns the .perfetto-trace bytes.

    mode:
      - "instant" (default): each record is emitted as a point-in-time marker
        (TYPE_INSTANT).  Milestones are honest: bubbles show up as empty gaps
        between points.
      - "slices": each track is fully covered by labelled bars: ``active``
        intervals and explicit ``stall: ...`` intervals around actual blocking
        waits. ``SIGNAL_*`` events are also retained as instant markers, so
        barrier signal -> receiver wake-up latency is visible on the same view.
    """
    if records.dtype != EVENT_DTYPE:
        records = np.asarray(records)
        cols = (
            records["ts"].astype("u8"),
            records["event"].astype("u4"),
            records["cta"].astype("u4"),
            records["role"].astype("u4"),
            records["iter"].astype("u4"),
        )
        records = np.rec.fromarrays(cols, dtype=EVENT_DTYPE)

    # One track per (cta, role) actually observed.
    keys = sorted(set(zip(records["cta"].tolist(), records["role"].tolist())))
    packets = []
    for cta, role in keys:
        uuid = int(cta) * 4 + int(role)
        role_name = ROLE_NAMES.get(int(role), f"role{role}")
        packets.append(_track_descriptor_packet(uuid, f"CTA{cta} {role_name}"))

    order = np.argsort(records["ts"], kind="stable")
    if mode == "instant":
        for idx in order:
            ts = int(records["ts"][idx])
            event = int(records["event"][idx])
            cta = int(records["cta"][idx])
            role = int(records["role"][idx])
            it = int(records["iter"][idx])
            uuid = cta * 4 + role
            packets.append(
                _track_event_packet(ts, uuid, TYPE_INSTANT, _event_label(event, it))
            )
        return _wrap_trace(packets)

    # slices: the complement of blocking-wait intervals is labelled active;
    # waits themselves are slices too, not anonymous gaps. This makes an idle
    # warpgroup visible even when the Perfetto UI compresses empty space.
    by_track = {}
    for idx in order:
        cta = int(records["cta"][idx])
        role = int(records["role"][idx])
        by_track.setdefault((cta, role), []).append(idx)

    # (ts, kind, uuid, name), kind in {"B", "E"}. At an adjacent boundary
    # ends must be emitted before begins, otherwise Perfetto treats them as
    # nested slices instead of consecutive states on one warpgroup track.
    staged = []
    signal_instants = []
    for (cta, role), idxs in by_track.items():
        uuid = cta * 4 + role
        bubbles = []                 # [bts, ets, name, begin_id, end_id]
        marks = []                   # (ts, name, event_id) work-phase end markers
        open_begins = {begin: [] for (begin, _) in BUBBLE_PAIRS}
        for idx in idxs:
            ts = int(records["ts"][idx])
            event = int(records["event"][idx])
            if event in SIGNAL_EVENTS or event in COMPLETION_EVENTS:
                signal_instants.append((
                    ts, uuid, event, int(records["iter"][idx])
                ))
            if event in open_begins:  # a bubble BEGIN
                open_begins[event].append(ts)
            elif event in _END_TO_BEGIN:
                begin = _END_TO_BEGIN[event]
                if open_begins[begin]:
                    bubbles.append((
                        open_begins[begin].pop(0), ts, BUBBLE_PAIRS[(begin, event)],
                        begin, event,
                    ))
            elif event in WORK_PHASES:
                marks.append((ts, WORK_PHASES[event], event))
        bubbles.sort(key=lambda x: (x[0], x[1]))
        marks.sort()
        lo = min(int(records["ts"][idx]) for idx in idxs)
        hi = max(int(records["ts"][idx]) for idx in idxs)
        # Build a non-overlapping state timeline. A work marker names the
        # active interval that ends at that marker; any unmarked interval is
        # still explicitly shown as active.
        def add_active(start, end, start_event=None, end_event=None):
            if end <= start:
                return
            inner_marks = [
                (mts, name, event) for mts, name, event in marks if start < mts <= end
            ]
            cursor = start
            active_event = start_event
            for mts, name, event in inner_marks:
                staged.append((cursor, "B", uuid, f"active: {name}"))
                staged.append((mts, "E", uuid, f"active: {name}"))
                cursor = mts
                active_event = event
            if cursor < end:
                name = ACTIVE_TRANSITIONS.get(
                    (active_event, end_event), "active (unannotated)"
                )
                staged.append((cursor, "B", uuid, f"active: {name}"))
                staged.append((end, "E", uuid, f"active: {name}"))

        cursor = lo
        cursor_event = None
        for bts, ets, name, begin_event, end_event in bubbles:
            add_active(cursor, bts, cursor_event, begin_event)
            if ets > bts:
                staged.append((bts, "B", uuid, f"stall: {name}"))
                staged.append((ets, "E", uuid, f"stall: {name}"))
            cursor = max(cursor, ets)
            cursor_event = end_event
        add_active(cursor, hi, cursor_event)

    # Packets in a trusted sequence must remain timestamp-ordered. In
    # particular, do not append signal instants after all duration bars: their
    # timestamps usually lie in the middle of the trace and Perfetto can then
    # hide or misorder them. At a shared timestamp close the old slice, show
    # the signal, then open the next slice.
    timeline = [
        (ts, 0 if kind == "E" else 2, "slice", kind, uuid, name)
        for ts, kind, uuid, name in staged
    ]
    timeline.extend(
        (ts, 1, "instant", event, uuid, it)
        for ts, uuid, event, it in signal_instants
    )
    timeline.sort()
    for ts, _order, kind, value, uuid, name_or_iter in timeline:
        if kind == "slice":
            packets.append(
                _track_event_packet(
                    ts, uuid,
                    TYPE_SLICE_BEGIN if value == "B" else TYPE_SLICE_END,
                    name_or_iter,
                )
            )
        else:
            packets.append(
                _track_event_packet(ts, uuid, TYPE_INSTANT, _event_label(value, name_or_iter))
            )

    return _wrap_trace(packets)


# --------------------------------------------------------------------------- #
# Minimal decoder, used only by --self-test to validate the encoder's wire   #
# format (structural round-trip, not a Perfetto-compatibility check).        #
# --------------------------------------------------------------------------- #
def _read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _decode_packet(pkt: bytes):
    pos = 0
    ts = None
    track_event = None
    track_descriptor = None
    while pos < len(pkt):
        tag, pos = _read_varint(pkt, pos)
        field = tag >> 3
        wire = tag & 7
        if wire == 0:
            val, pos = _read_varint(pkt, pos)
            if field == 8:
                ts = val
        elif wire == 2:
            ln, pos = _read_varint(pkt, pos)
            sub = pkt[pos:pos + ln]
            pos += ln
            if field == 11:
                track_event = sub
            elif field == 60:
                track_descriptor = sub
        else:
            raise ValueError(f"unsupported wire type {wire}")
    return ts, track_event, track_descriptor


def _decode_track_event(te: bytes):
    pos = 0
    uuid = None
    type_ = None
    name = None
    while pos < len(te):
        tag, pos = _read_varint(te, pos)
        field = tag >> 3
        wire = tag & 7
        if wire == 0:
            val, pos = _read_varint(te, pos)
            if field == 11:
                uuid = val
            elif field == 9:
                type_ = val
        elif wire == 2:
            ln, pos = _read_varint(te, pos)
            sub = te[pos:pos + ln]
            pos += ln
            if field == 23:
                name = sub.decode()
        else:
            raise ValueError("bad wire type in TrackEvent")
    return uuid, type_, name


def _decode_track_descriptor(td: bytes):
    pos = 0
    uuid = None
    name = None
    thread = None
    while pos < len(td):
        tag, pos = _read_varint(td, pos)
        field = tag >> 3
        wire = tag & 7
        if wire == 0:
            val, pos = _read_varint(td, pos)
            if field == 1:
                uuid = val
        elif wire == 2:
            ln, pos = _read_varint(td, pos)
            sub = td[pos:pos + ln]
            pos += ln
            if field == 2:
                name = sub.decode()
            elif field == 4:
                thread = sub
        else:
            raise ValueError("bad wire type in TrackDescriptor")
    return uuid, name, thread


def _decode_all(blob: bytes):
    """Decode a full trace blob into (n_descriptors, list[(type, name)])."""
    pos = 0
    n_descriptors = 0
    events = []
    while pos < len(blob):
        tag, pos = _read_varint(blob, pos)
        assert tag == (1 << 3) | 2, f"expected Trace.packet tag, got {tag}"
        ln, pos = _read_varint(blob, pos)
        pkt = blob[pos:pos + ln]
        pos += ln
        _ts, track_event, track_descriptor = _decode_packet(pkt)
        if track_descriptor is not None:
            uuid, name, thread = _decode_track_descriptor(track_descriptor)
            assert uuid is not None, "TrackDescriptor missing uuid (field 1)"
            assert name is not None, "TrackDescriptor missing name (field 2)"
            assert thread is not None, "TrackDescriptor missing thread (field 4)"
            n_descriptors += 1
        elif track_event is not None:
            uuid, type_, name = _decode_track_event(track_event)
            assert uuid is not None, "TrackEvent missing track_uuid (field 11)"
            assert name is not None, "TrackEvent missing name (field 23)"
            events.append((type_, name))
    return n_descriptors, events


def _self_test() -> None:
    # 8 records across two tracks:
    #   consumer (role 1): K-wait bubble 16->17 [100,150], work "QK^T GEMM"
    #     ending at 10@200, sM-wait bubble 18->19 [210,230], work "softmax"
    #     ending at 11@260;
    #   producer (role 0): wait-free bubble 2->22 [100,150] (no work markers).
    records = np.rec.fromarrays(
        [
            np.array([100, 150, 200, 210, 230, 260, 100, 150], dtype="u8"),
            np.array([16, 17, 10, 18, 19, 11, 2, 22], dtype="u4"),
            np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype="u4"),
            np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype="u4"),
            np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype="u4"),
        ],
        dtype=EVENT_DTYPE,
    )

    # instant mode: 8 point markers.
    n_desc, events = _decode_all(build_perfetto_trace(records, mode="instant"))
    assert n_desc == 2, n_desc  # tracks (cta0,role1) and (cta0,role0)
    assert len(events) == 8 and all(t == TYPE_INSTANT for t, _ in events), events

    # slices mode: both active and stall intervals are labelled. The producer
    # is included even though it has no consumer compute milestone.
    n_desc, events = _decode_all(build_perfetto_trace(records, mode="slices"))
    assert n_desc == 2, n_desc
    begins = [n for t, n in events if t == TYPE_SLICE_BEGIN]
    ends = [n for t, n in events if t == TYPE_SLICE_END]
    expected = {
        "stall: wait K", "active: QK^T GEMM", "stall: wait sM",
        "active: softmax", "active: mask scores", "stall: wait V0L free",
    }
    assert set(begins) == expected and len(begins) == len(expected), begins
    assert set(ends) == expected and len(ends) == len(expected), ends
    assert sum(t == TYPE_INSTANT for t, _ in events) == 0, events
    assert len(events) == 12, len(events)
    print(f"self-test: OK ({n_desc} descriptors, instant + active/stall-slice round-trip)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", help=".npz file with fields ts,event,cta,role,iter")
    ap.add_argument("--out", default="trace.perfetto-trace", help="output trace path")
    ap.add_argument(
        "--mode",
        choices=("slices", "instant"),
        default="instant",
        help="instant = point markers (default, bubbles show as gaps); "
        "slices = work/compute as filled bars, waits as gaps",
    )
    ap.add_argument("--self-test", action="store_true", help="round-trip the encoder")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.events:
        ap.error("provide --events (or use --self-test)")

    with np.load(args.events) as z:
        records = np.rec.fromarrays(
            [
                z["ts"].astype("u8"),
                z["event"].astype("u4"),
                z["cta"].astype("u4"),
                z["role"].astype("u4"),
                z["iter"].astype("u4"),
            ],
            dtype=EVENT_DTYPE,
        )
    blob = build_perfetto_trace(records, mode=args.mode)
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"wrote {len(blob)} bytes to {args.out} (open at https://ui.perfetto.dev)")


if __name__ == "__main__":
    sys.exit(main())
