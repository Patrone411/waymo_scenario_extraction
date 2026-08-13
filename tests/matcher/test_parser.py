from pathlib import Path
import os

from osc2_parser.parser import OSCProgram


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_EXTRACTION = REPO_ROOT / "scenario_extraction"

OSC_FILE = (
    SCENARIO_EXTRACTION
    / "osc2_parser"
    / "osc"
    / "change_lane.osc"
)


def test_osc_file_parses_without_error():
    assert OSC_FILE.exists(), (
        f"Test-OSC file not found: {OSC_FILE}"
    )

    old_cwd = Path.cwd()

    try:
        os.chdir(SCENARIO_EXTRACTION)

        prog = OSCProgram(
            osc_path="osc2_parser/osc/change_lane.osc"
        ).compile()
        
        """calls = []
        dur_by_label = getattr(prog, "block_durations", {}) or {}
        for c in prog.calls:
            c2  = dict(c)
            lbl = c2.get("block_label")
            if lbl in dur_by_label and c2.get("duration") is None:
                c2["block_duration"] = dur_by_label[lbl]
            c2.setdefault("block_overlap", "start")
            calls.append(c2)"""

        assert prog is not None

    finally:
        os.chdir(old_cwd)