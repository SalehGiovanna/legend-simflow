# ruff: noqa: I002

# Copyright (C) 2025 Luigi Pertoldi <gipert@pm.me>,
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import awkward as ak
import legenddataflowscripts as ldfs
import legenddataflowscripts.utils
import numpy as np
from dbetto import AttrsDict
from dbetto.utils import load_dict
from lgdo import Array, Table, VectorOfVectors, lh5

from legendsimflow import nersc, patterns, spms_pars, utils
from legendsimflow import reboost as reboost_utils
from legendsimflow.awkward import ak_isin
from legendsimflow.metadata import encode_usability
from legendsimflow.profile import make_profiler
from legendsimflow.tcm import merge_stp_n_opt_tcms_to_lh5

GEDS_ENERGY_THR_KEV = 25
SPMS_ENERGY_THR_PE = 0
BUFFER_LEN = "50*MB"
OFF = encode_usability("off")
ON = encode_usability("on")

args = nersc.dvs_ro_snakemake(snakemake)  # noqa: F821

wildcards = args.wildcards
jobid = wildcards.jobid
stp_file = patterns.output_simjob_filename(
    args.config, tier="stp", simid=wildcards.simid, jobid=jobid
)
hit_file = {
    "opt": args.input.opt_file,
    "hit": args.input.hit_file,
}
evt_file = args.output[0]
log_file = args.log[0]
metadata = args.config.metadata
simstat_part_file = args.input.simstat_part_file
add_random_coincidences = args.params.add_random_coincidences
l200data = args.config.paths.l200data
usabilities = AttrsDict(load_dict(args.input.detector_usabilities[0]))

evt_file, move2cfs = nersc.make_on_scratch(args.config, evt_file)

# setup logging
log = ldfs.utils.build_log(metadata.simprod.config.logging, log_file)
perf_block, print_perf, _ = make_profiler()

log.info("merging hit and opt TCMs")
with perf_block("merge_tcms()"):
    scintillator_uid = next(
        uid
        for uid, name in reboost_utils.get_remage_detector_uids(stp_file).items()
        if name == "liquid_argon"
    )

    merge_stp_n_opt_tcms_to_lh5(
        stp_file,
        hit_file["opt"],
        evt_file,
        scintillator_uid=scintillator_uid,
        buffer_len=BUFFER_LEN,
    )


# test that the evt tcm has the same amount of rows as the stp tcm

if lh5.read_n_rows("tcm", stp_file) != lh5.read_n_rows("tcm", evt_file):
    msg = (
        "stp and evt tcm should have same number of rows not "
        f"stp={lh5.read_n_rows('tcm', stp_file)}, "
        f"evt={lh5.read_n_rows('tcm', evt_file)}, "
        f"hit={lh5.read_n_rows('tcm', hit_file['hit'])}, "
        f"opt={lh5.read_n_rows('tcm', hit_file['opt'])}"
    )

    raise ValueError(msg)

# get the mapping of detector name to uid
# NOTE: we check on disk because we are not sure which tables were processed in
# the hit tiers
det2uid = {}
for tier in ("opt", "hit"):
    det2uid[tier] = {
        name: uid
        for uid, name in reboost_utils.get_remage_detector_uids(
            hit_file[tier], lh5_table="hit"
        ).items()
    }
    msg = f"found mapping name -> uid ({tier} tier): {det2uid[tier]}"
    log.debug(msg)


# little helper to simplify the code below
# TODO: move/fix in reboost
def _read_hits(tcm_ak, tier, field):
    msg = f"loading {field=} data from {tier=} (file {hit_file[tier]})"
    log.debug(msg)

    with perf_block("read_hits()"):
        tcm = tcm_ak[tier]
        with perf_block("flattening tcm"):
            tcm_flat = ak.Array({k: ak.flatten(tcm[k]) for k in tcm.fields})

        data_flat = []
        tcm_rows = []

        # for un-flattening at the end
        counts = ak.num(tcm.row_in_table)

        for tab_name, key in det2uid[tier].items():
            mask = tcm_flat.table_key == key

            with perf_block("filtering row_in_table"):
                rows = tcm_flat.row_in_table[mask].to_numpy()
                tcm_rows.append(np.where(mask)[0].to_numpy())

            with perf_block("lh5.read()"):
                data_ch = lh5.read(f"hit/{tab_name}/{field}", hit_file[tier], idx=rows)

            units = data_ch.attrs.get("units", None)
            data_ch = data_ch.view_as("ak")

            data_flat.append(data_ch)

        tcm_rows_concat = np.concatenate(tcm_rows)
        data_flat_concat = ak.concatenate(data_flat)[np.argsort(tcm_rows_concat)]

        data_unflat = ak.unflatten(data_flat_concat, counts)

        if units is not None:
            return ak.with_parameter(data_unflat, "units", units)
        return data_unflat


partitions = load_dict(simstat_part_file)[f"job_{jobid}"]

# use write_safe on the first chunk to catch stale data from a failed retry
evt_wo_mode = "write_safe"

log.info("begin iterating over TCM")
for runid_idx, (runid, evt_idx_range) in enumerate(partitions.items()):
    msg = (
        f"processing partition corresponding to {runid} "
        f"[{runid_idx + 1}/{len(partitions)}], event range {evt_idx_range}"
    )
    log.info(msg)

    evt_start, evt_end = evt_idx_range
    # evt_idx_range is [start, end] inclusive
    n_entries = evt_end - evt_start + 1

    # canonical non-OFF SiPM channel UIDs for this run, in ascending order.
    # used to pad events with no SiPM photons so rawid/energy/time always
    # carry the full channel dimension (empty arrays for inactive channels)
    canonical_spms_uids = sorted(
        uid
        for det_name, uid in det2uid["opt"].items()
        if usabilities[runid].get(det_name, "on") != "off"
    )

    if add_random_coincidences:
        msg = "looking up forced trigger files for random coincidences"
        log.debug(msg)
        with perf_block("lookup_rc_files()"):
            evt_tier_name = utils.get_evt_tier_name(l200data)
            rc_evt_files = sorted(
                spms_pars.lookup_evt_files(l200data, runid, evt_tier_name)
            )
            if not rc_evt_files:
                msg = "no RC evt files found for random coincidences"
                raise RuntimeError(msg)
        with perf_block("build_rc_evt_index_lookup()"):
            rc_index_lookup = spms_pars.build_rc_evt_index_lookup(rc_evt_files)
        # state is reset per partition so RC events are drawn independently
        # for each run slice
        rc_file_state: dict = {}

    # iterate over the unified tcm for this partition; an empty partition
    # (n_entries=0) produces no chunks and is silently skipped
    # NOTE: open mode is append because we will write to the same file
    it = lh5.LH5Iterator(
        str(evt_file),
        "tcm",
        i_start=evt_start,
        n_entries=n_entries,
        buffer_len=BUFFER_LEN,
        h5py_open_mode="a",
    )
    for chunk in it:
        unified_tcm = chunk.view_as("ak")
        out_table = Table(size=len(unified_tcm))

        # split the unified TCM in two, one for each tier. in this way we will be
        # able to read data from each tier
        tcm = {}
        for tier in ("opt", "hit"):
            mask = ak_isin(unified_tcm.table_key, det2uid[tier].values())
            tcm[tier] = unified_tcm[mask]

        # trigger table
        # -------------
        out_table.add_field("trigger", Table(size=len(unified_tcm)))

        # global fields that are constant over the full events
        # let's take them from the hit tier
        for constant_field in ["run", "period", "evtid"]:
            data = _read_hits(tcm, "hit", constant_field)

            # sanity check
            assert len(data) == len(tcm["hit"])

            # replace the awkward missing values with NaN for LH5 compatibility
            data = ak.fill_none(ak.firsts(data, axis=-1), np.nan)
            out_table.add_field(f"trigger/{constant_field}", Array(data))

        timestamp = _read_hits(tcm, "hit", "t0")
        timestamp = ak.fill_none(ak.firsts(timestamp, axis=-1), np.nan)
        out_table.add_field(
            "trigger/timestamp", Array(timestamp, attrs={"units": "ns"})
        )

        # HPGe table
        # ----------
        out_table.add_field("geds", Table(size=len(unified_tcm)))

        # first read usability and energy
        usability = _read_hits(tcm, "hit", "usability")
        energy = _read_hits(tcm, "hit", "energy")

        # we want to only store hits from events in ON and AC detectors and above
        # our energy threshold
        hitsel = (usability != OFF) & (energy > GEDS_ENERGY_THR_KEV)

        # we want to still be able to know which detectors are ON (and not AC)
        out_table.add_field(
            "geds/is_good_channel", VectorOfVectors(usability[hitsel] == ON)
        )
        out_table.add_field(
            "geds/energy", VectorOfVectors(energy[hitsel], attrs={"units": "keV"})
        )
        # NOTE: the energy sum does not include AC detectors
        out_table.add_field(
            "geds/energy_sum",
            Array(
                ak.sum(energy[hitsel & (usability == ON)], axis=-1),
                attrs={"units": "keV"},
            ),
        )

        # fields to identify detectors and lookup stuff in the lower tiers
        out_table.add_field("geds/rawid", VectorOfVectors(tcm["hit"].table_key[hitsel]))
        out_table.add_field(
            "geds/hit_idx", VectorOfVectors(tcm["hit"].row_in_table[hitsel])
        )

        # simply forward some fields
        aoe = _read_hits(tcm, "hit", "aoe")
        out_table.add_field("geds/aoe", VectorOfVectors(aoe[hitsel]))
        out_table.add_field("geds/has_aoe", VectorOfVectors(~np.isnan(aoe[hitsel])))

        is_ss = _read_hits(tcm, "hit", "is_single_site")
        out_table.add_field("geds/is_single_site", VectorOfVectors(is_ss[hitsel]))

        # compute multiplicity
        geds_multiplicity = ak.sum(hitsel, axis=-1)
        out_table.add_field("geds/multiplicity", Array(geds_multiplicity))

        # SiPM table
        # ----------
        out_table.add_field("spms", Table(size=len(unified_tcm)))

        # also here, we exclude the non usable channels. this is in line with what
        # done in the evt tier in pygama
        usability = _read_hits(tcm, "opt", "usability")
        energy = _read_hits(tcm, "opt", "energy")
        chansel = usability != OFF
        # we also discard all pulses with amplitude below threshold
        pesel = energy > SPMS_ENERGY_THR_PE

        # in simulation the opt TCM only records channels that detected
        # photons, so events with no SiPM activity have empty arrays.
        # pad those events with the canonical non-OFF channel list (empty
        # PE arrays) to match the real-data convention where all non-OFF
        # channels are always present.
        # NOTE: the canonical_spms_uids ordering must match the ordering
        # used by non-empty events (i.e. the TCM ordering). Currently both
        # are ascending by UID. The implicit assumption for RC data (added
        # below) is that RC evt files come from runs whose usability map
        # matches the current run partition.
        n_events = len(unified_tcm)
        is_empty_opt = ak.num(tcm["opt"].table_key) == 0
        canonical_broadcast = ak.Array([canonical_spms_uids] * n_events)

        rawid = tcm["opt"].table_key[chansel]
        rawid = ak.where(is_empty_opt, canonical_broadcast, rawid)
        out_table.add_field("spms/rawid", VectorOfVectors(rawid))

        energy_sel = energy[pesel][chansel]
        empty_energy = ak.Array([[[] for _ in canonical_spms_uids]] * n_events)
        energy_sel = ak.where(is_empty_opt, empty_energy, energy_sel)
        out_table.add_field("spms/energy", VectorOfVectors(energy_sel))

        is_saturated = _read_hits(tcm, "opt", "is_saturated")
        is_saturated_sel = is_saturated[chansel]
        empty_is_saturated = ak.Array([[False for _ in canonical_spms_uids]] * n_events)
        is_saturated_sel = ak.where(is_empty_opt, empty_is_saturated, is_saturated_sel)
        out_table.add_field("spms/is_saturated", VectorOfVectors(is_saturated_sel))

        hit_idx = tcm["opt"].row_in_table[chansel]
        empty_hit_idx = ak.Array([[-1 for _ in canonical_spms_uids]] * n_events)
        hit_idx = ak.where(is_empty_opt, empty_hit_idx, hit_idx)
        out_table.add_field("spms/hit_idx", VectorOfVectors(hit_idx))

        time = _read_hits(tcm, "opt", "time")
        time_sel = time[pesel][chansel]
        empty_time = ak.Array([[[] for _ in canonical_spms_uids]] * n_events)
        time_sel = ak.where(is_empty_opt, empty_time, time_sel)
        out_table.add_field(
            "spms/time", VectorOfVectors(time_sel, attrs={"units": "ns"})
        )

        if add_random_coincidences:
            with perf_block("get_chunk_rc_data()"):
                rc_chunk = spms_pars.get_chunk_rc_data(
                    [str(f) for f in rc_evt_files],
                    rc_file_state,
                    len(unified_tcm),
                    rc_index_lookup,
                )
            # assert rawid alignment: RC and simulation must use the same
            # channel ordering (both are ascending by UID)
            assert ak.to_list(rc_chunk.rawid[0]) == canonical_spms_uids, (
                "RC rawid does not match simulation spms/rawid — "
                "check that RC evt files come from a run with the same "
                "usability map as the current run partition"
            )
            out_table.add_field("spms/rc_energy", VectorOfVectors(rc_chunk.npe))
            out_table.add_field(
                "spms/rc_time", VectorOfVectors(rc_chunk.t0, attrs={"units": "ns"})
            )

        # total amount of light per event
        energy_sum = ak.sum(ak.sum(energy[pesel][chansel], axis=-1), axis=-1)
        out_table.add_field("spms/energy_sum", Array(energy_sum))

        # how many channels saw some light
        spms_multiplicity = ak.sum(ak.any(chansel & pesel, axis=-1), axis=-1)
        out_table.add_field("spms/multiplicity", Array(spms_multiplicity))

        # coincidences table
        # ------------------
        out_table.add_field("coincident", Table(size=len(unified_tcm)))

        # is there a signal in the HPGe array?
        out_table.add_field("coincident/geds", Array(geds_multiplicity > 0))

        # is there a signal in the LAr instrumentation?
        lar_veto = (spms_multiplicity >= 4) | (energy_sum >= 4)
        out_table.add_field("coincident/spms", Array(lar_veto))

        # now write down
        with perf_block("write_chunk()"):
            lh5.write(out_table, "evt", evt_file, wo_mode=evt_wo_mode)
            evt_wo_mode = "append"

with perf_block("move_to_cfs()"):
    move2cfs()


print_perf()
