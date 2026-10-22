r"""Convert a person image to the standard VACE appearance-reference format.

Required: --input IMAGE --output PNG. Default: 832x480, RGB(127,127,127),
fixed aspect ratio, full person within the standard template's person bounds.
The default template is person-assets/references/traffic-officer-reference.png
when available; --no-template uses a centered 90%-size canvas instead.

--background-mode auto uses alpha first, otherwise removes border-connected
blue or solid backgrounds. Complex backdrops need a transparent PNG or --mask
(white=person, black=background). --background-mode keep preserves the backdrop.
No pose conversion or person synthesis is performed; no CSV is changed.

Examples:
  python tools/convert_person_reference.py --input /path/to/person.png \
    --output person-assets/references/person-reference.png
  python tools/convert_person_reference.py --input /path/to/blue-screen.jpg \
    --background-mode blue --output outputs/person-reference.png
  python tools/convert_person_reference.py --input /path/to/photo.jpg \
    --mask /path/to/person-mask.png --output outputs/person-reference.png
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.person_reference import convert_reference


def main(argv=None):
    standard = ROOT / "person-assets/references/traffic-officer-reference.png"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--template", type=Path, default=standard if standard.is_file() else None)
    group.add_argument("--no-template", action="store_true")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--background-mode", choices=("auto", "alpha", "blue", "solid", "keep"), default="auto")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--occupancy", type=float, default=.9, help="Person fit area without a template")
    parser.add_argument("--background", type=int, nargs=3, default=[127, 127, 127])
    parser.add_argument("--tolerance", type=float, default=24, help="Solid-background RGB distance threshold")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = convert_reference(
            args.input, args.output, template=None if args.no_template else args.template,
            mask=args.mask, mode=args.background_mode, width=args.width, height=args.height,
            occupancy=args.occupancy, background=args.background, tolerance=args.tolerance,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Reference conversion error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
