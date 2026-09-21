import os
import cv2
import argparse
import template_match 

# Text overlay settings
_MARKER_COLOR = (255, 0, 255)
_TEXT_COLOR   = (90, 90, 90)
_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE   = 0.5
_THICK        = 2
_offset       = {"y": -30}


def place_text(text, pt, img):
    """
    Draw text beside a point, placing it above the point if there's room,
    otherwise below.  Background rectangle matches exactly the text size.
    """
    # measure text
    (w, h), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICK)
    margin = 5

    # choose x so we don't overflow right edge
    x = pt[0] + margin if pt[0] + w + margin < img.shape[1] else pt[0] - w - margin

    # try to place above the point
    y_above = pt[1] - margin
    if y_above - h >= 0:
        y = y_above
    else:
        # not enough room above, place below
        y = pt[1] + h + margin

    # draw tight background rectangle
    top_left     = (x,       y - h    )
    bottom_right = (x + w,   y + baseline)
    cv2.rectangle(img, top_left, bottom_right, (255,255,255), cv2.FILLED)

    # finally draw the text
    cv2.putText(img, text, (x, y), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICK)


def run_one(scene_path, tmpl_path, label, output_dir):
    """
    Perform all four matching methods on the given scene and template,
    draw a rectangle and label at each match, then save the results.
    """
    # load inputs
    img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: could not load scene image '{scene_path}'")
        return
    if tmpl is None:
        print(f"Error: could not load template image '{tmpl_path}'")
        return

    methods = ["tm_ssd", "tm_nssd", "tm_ccor", "tm_nccor"]

    for method in methods:
        pt = template_match.template_match(img, tmpl, method)
        br = (pt[0] + tmpl.shape[1], pt[1] + tmpl.shape[0])
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(out, pt, br, _MARKER_COLOR, 2)
        place_text(f"{pt}", pt, out)

        out_filename = f"{method}-{label}.png"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, out)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run template matching on one scene/template pair."
    )
    parser.add_argument("scene",     help="path to the scene image (PNG)")
    parser.add_argument("template",  help="path to the template image (PNG)")
    parser.add_argument("label",     help="label to include in output filenames")
    parser.add_argument(
        "--output_dir",
        default="output",
        help="directory in which to write annotated results (default: ./output)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_one(args.scene, args.template, args.label, args.output_dir)


if __name__ == "__main__":
    main()
