"""Publish compact audit scores with one shared copy of each reference window."""

import json
from pathlib import Path
from sys import argv


def compact_panel(report, ref_name, activity_name):
    activity = report[activity_name]
    return {
        **report["exact_three_detection"][ref_name],
        "all_DER": activity["full"]["DER"],
        "reference_speaker_s": activity["full"]["reference_speaker_s"],
        "single_DER": activity["count_1"]["DER"],
        "double_DER": activity["count_2"]["DER"],
        "triple_speaker_recall": activity["count_3"]["speaker_recall"],
        "exact_set_fraction_by_window": [
            round(window["exact_set_fraction"], 6)
            for window in report["exact_three_windows"][ref_name]
        ],
    }


def compact(source):
    output = {}
    for meeting, variants in source.items():
        windows = next(iter(variants.values()))["exact_three_windows"]
        reference = {
            name: [[x["start_s"], x["end_s"]] for x in items]
            for name, items in windows.items()
        }
        predictions = {}
        output[meeting] = dict(reference_exact_three_windows_s=reference, predictions=predictions)
        for model, report in variants.items():
            row = predictions[model] = dict(
                map_system_to_reference=report["activity_mapping"],
                map_policy=report["mapping_policy"],
                RTTM=compact_panel(report, "RTTM", "RTTM_activity"),
                TextGrid=compact_panel(report, "TextGrid", "alternate_TextGrid_activity"),
            )
            if "cpCER" in report:
                cpcer = report["cpCER"]
                row["cpCER_edits"] = cpcer["edits"]
                row["cpCER_ref_chars"] = cpcer["reference_chars"]
                row["fixed_map_edits"] = report["text_fixed_activity_mapping"]["edits"]
                row["timed"] = {
                    key: {
                        "all_edits": data["full"]["edits"],
                        "triple_edits": data["count_3"]["edits"],
                        "triple_ref_chars": data["count_3"]["reference_chars"],
                        "triple_exact_reference_times": data["count_3"]["supplied_reference_char_timestamps"],
                    }
                    for key, data in report["timed"].items()
                }
    return output


if __name__ == "__main__":
    source, destination = map(Path, argv[1:])
    results = compact(json.loads(source.read_text(encoding="utf-8")))
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
