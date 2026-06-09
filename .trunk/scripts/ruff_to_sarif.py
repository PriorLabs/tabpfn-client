#!/usr/bin/env python3

import json
import sys


results = []


def get_region(entry, column_offset=0):
    location = entry["location"]
    region = {
        "startColumn": location["column"] + column_offset,
        "startLine": location["row"],
    }
    if "end_location" in entry:
        end_location = entry["end_location"]
        region["endColumn"] = end_location["column"] + column_offset
        region["endLine"] = end_location["row"]
    return region


# As of ruff v0.0.266, column edits are 1-indexed. This is handled by the command definition
ruff_column_index = 1
if len(sys.argv) > 1:
    ruff_column_index = int(sys.argv[1])

for result in json.load(sys.stdin):
    # As of ruff v0.0.260, some autofixable diagnostics may appear redundantly
    if "location" not in result:
        continue

    filepath = result["filename"]
    # Ruff will set code to null for syntax errors
    rule_id = result["code"] or "E999"
    message = result["message"]

    sarif_result = {
        "level": "error",
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": filepath,
                    },
                    "region": get_region(result),
                }
            }
        ],
        "message": {
            "text": message,
        },
        "ruleId": rule_id,
    }

    if "fix" in result and result["fix"] is not None:
        fixes = result["fix"]

        # TODO(Tyler): If output format changes any more substantially, consider version-specific parsers
        # As of ruff v0.0.260, autofixes are nested inside fix.edits
        if "edits" in fixes:
            message = fixes["message"]
            fixes = fixes["edits"]
        else:
            message = fixes["message"]
            fixes = [fixes]

        sarif_result["fixes"] = [
            {
                "description": {
                    "text": message,
                },
                "artifactChanges": [
                    {
                        "artifactLocation": {
                            "uri": filepath,
                        },
                        "replacements": [
                            {
                                # Ruff gives 0-indexed columns, SARIF requires 1-indexed
                                # https://github.com/charliermarsh/ruff/issues/3106
                                "deletedRegion": get_region(fix, ruff_column_index),
                                "insertedContent": {
                                    "text": fix["content"],
                                },
                            }
                        ],
                    }
                ],
            }
            for fix in fixes
        ]

    results.append(sarif_result)

sarif = {
    "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
    "version": "2.1.0",
    "runs": [{"results": results}],
}

print(json.dumps(sarif, indent=2))
