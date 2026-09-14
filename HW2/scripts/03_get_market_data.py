"""Step 3: download the four indicators and the 3-month bill control; build the daily panel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.market import build_panel

if __name__ == "__main__":
    panel = build_panel()
    print(panel.tail())
    print(panel.notna().sum())
