"""
Merge the Champions League fixtures from the wide-window fixture pull into the
main fixture file.

`acquire_next_matches.py --window_days 7` leaves CL empty whenever the next
European matchday is more than a week away, but the domestic leagues must stay
on the 7-day window so that every fixture still has a Flashscore odds feed.

Usage:
    python scripts/merge_cl.py
"""
import os
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(BASE, "data", "next_matches.json")
WIDE = os.path.join(BASE, "data", "next_matches_16d.json")


def load(path):
    with open(path, encoding="utf-16") as f:
        return json.load(f)


def main():
    if not os.path.exists(WIDE):
        print("No wide-window fixture file, nothing to merge.")
        return
    main_data, wide = load(MAIN), load(WIDE)
    cl_main = main_data.get("CL", {}).get("next_matches", [])
    cl_wide = wide.get("CL", {}).get("next_matches", [])
    if cl_main:
        print(f"CL already has {len(cl_main)} fixtures, left untouched.")
        return
    if not cl_wide:
        print("No CL fixtures in the wide window either.")
        return
    main_data["CL"]["next_matches"] = cl_wide
    with open(MAIN, "w", encoding="utf-16") as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(cl_wide)} Champions League fixtures.")


if __name__ == "__main__":
    main()
