"""Capture diverse, user-selected person-dataset frames from a local webcam."""
import argparse
from pathlib import Path
import re
import sys
import cv2


def _session_name(value):
    if value is None: return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+",value):
        raise ValueError("--session may contain only letters, numbers, underscore, and hyphen")
    return value


def camera_smoke_test(camera_index):
    """Open camera and read one frame, always releasing the device."""
    cap=None
    try:
        cap=cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False, f"Camera index {camera_index} could not be opened by OpenCV. Try --camera 1 (or another available index), check camera permissions, and close apps currently using the webcam."
        ok,frame=cap.read()
        if not ok or frame is None:
            return False, f"Camera index {camera_index} opened but OpenCV could not read a frame. Try --camera 1 (or another available index) and check camera permissions."
        h,w=frame.shape[:2]
        return True, f"Camera index {camera_index} opened and returned a {w}x{h} frame. No image was saved."
    except cv2.error as exc:
        return False, f"OpenCV error for camera index {camera_index}: {exc}. Try --camera 1 (or another available index)."
    finally:
        if cap is not None: cap.release()


def _next_filename(output_dir, session):
    prefix=f"{session}_" if session else "person_cam_"
    number=1
    pattern=re.compile(re.escape(prefix)+r"(\d{6})\.jpg$")
    for path in output_dir.glob(f"{prefix}*.jpg"):
        match=pattern.fullmatch(path.name)
        if match: number=max(number,int(match.group(1))+1)
    while True:
        candidate=output_dir/f"{prefix}{number:06d}.jpg"
        if not candidate.exists(): return candidate
        number+=1


def _draw_status(frame, frame_number, captured, camera_index, output_dir, session, paused):
    preview=frame.copy()
    lines=[f"Camera {camera_index} | frame {frame_number} | captured this run: {captured}",
           f"Session: {session or 'none'} | save folder: {output_dir}",
           "SPACE capture | P pause/resume | D review last | Q quit"]
    if paused: lines.append("PAUSED — P to resume")
    y=28
    for line in lines:
        cv2.putText(preview,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.58,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(preview,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),1,cv2.LINE_AA)
        y+=27
    return preview


def _review_last(path):
    image=cv2.imread(str(path))
    if image is None: return
    cv2.putText(image,"LAST CAPTURE — any key returns (Q quits)",(12,28),cv2.FONT_HERSHEY_SIMPLEX,
                0.65,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("Person dataset capture — review",image)
    key=cv2.waitKey(0)&0xFF
    cv2.destroyWindow("Person dataset capture — review")
    return key in (ord("q"),ord("Q"),27)


def capture(camera_index=0, output_dir="data/person/images", session=None):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    cap=None; captured=0; frame_number=0; last_saved=None; paused=False; frozen=None
    try:
        cap=cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"ERROR: camera index {camera_index} could not be opened by OpenCV. Try --camera 1 (or another available index), check camera permissions, and close apps currently using the webcam.",file=sys.stderr)
            return 2
        ok,first=cap.read()
        if not ok or first is None:
            print(f"ERROR: camera index {camera_index} opened but no frame could be read. Try --camera 1 (or another available index).",file=sys.stderr)
            return 2
        frozen=first
        window="Person dataset capture"
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        while True:
            if not paused:
                ok,frame=cap.read()
                if not ok or frame is None:
                    print(f"ERROR: camera index {camera_index} stopped returning frames.",file=sys.stderr)
                    return 2
                frozen=frame; frame_number+=1
            preview=_draw_status(frozen,frame_number,captured,camera_index,output_dir,session,paused)
            cv2.imshow(window,preview)
            key=cv2.waitKey(30 if not paused else 100)&0xFF
            if key in (ord("q"),ord("Q"),27): break
            if key in (ord("p"),ord("P")): paused=not paused
            elif key==ord(" ") and not paused:
                path=_next_filename(output_dir,session)
                if cv2.imwrite(str(path),frozen,[cv2.IMWRITE_JPEG_QUALITY,95]):
                    captured+=1; last_saved=path
                    print(f"Saved {path}")
                else:
                    print(f"ERROR: OpenCV could not save {path}",file=sys.stderr)
            elif key in (ord("d"),ord("D")) and last_saved is not None:
                if _review_last(last_saved): break
            # Capturing only on an explicit key avoids saving every webcam frame.
        print(f"Capture ended. Saved {captured} new image(s) in {output_dir}.")
        return 0
    except cv2.error as exc:
        print(f"OpenCV error for camera index {camera_index}: {exc}. Try --camera 1 (or another available index).",file=sys.stderr)
        return 2
    finally:
        if cap is not None: cap.release()
        try: cv2.destroyAllWindows()
        except cv2.error: pass


def main(argv=None):
    parser=argparse.ArgumentParser(description="Capture manually selected real frames for the custom person dataset; does not annotate or train.")
    parser.add_argument("--camera",type=int,default=0,help="OpenCV camera index (default: 0)")
    parser.add_argument("--session",type=_session_name,help="Session prefix for filenames, e.g. session_01")
    parser.add_argument("--output-dir",default="data/person/images",help="Capture directory")
    parser.add_argument("--test-only",action="store_true",help="Check that camera opens and returns a frame, without saving or opening a window")
    args=parser.parse_args(argv)
    if args.camera<0: parser.error("--camera must be a non-negative index")
    if args.test_only:
        ok,message=camera_smoke_test(args.camera)
        print(message,file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 2
    return capture(args.camera,args.output_dir,args.session)


if __name__=="__main__": raise SystemExit(main())
