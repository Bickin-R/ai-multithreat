"""Capture explicitly selected safe-prop candidate frames; never writes final manifests."""
import argparse
from pathlib import Path
import cv2
from .capture_person_webcam import _session_name, _next_filename


def _preview(frame,frame_number,captured,camera_index,output_dir,session,paused):
    image=frame.copy()
    lines=[f"Camera {camera_index} | frame {frame_number} | captured this run: {captured}",
           f"Session: {session} | candidate inbox: {output_dir}",
           "SAFE TRAINING PROPS ONLY — do not handle functional weapons",
           "SPACE capture | P pause/resume | Q quit"]
    if paused: lines.append("PAUSED — press P to resume")
    y=28
    for line in lines:
        color=(0,0,255) if line.startswith("SAFE") else (255,255,255)
        cv2.putText(image,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(image,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.58,color,1,cv2.LINE_AA)
        y+=28
    return image


def capture(camera_index=0, session="weapon_session_01", output_dir="data/weapon/candidates/inbox"):
    session=_session_name(session)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    cap=None; captured=0; frame_number=0; paused=False; frozen=None
    try:
        cap=cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"ERROR: camera index {camera_index} could not be opened. Check camera permissions or try another index."); return 2
        ok,frozen=cap.read()
        if not ok or frozen is None: print(f"ERROR: camera index {camera_index} returned no frame."); return 2
        window="Weapon candidate capture — safe props only"; cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        while True:
            if not paused:
                ok,frame=cap.read()
                if not ok or frame is None: print("ERROR: webcam stopped returning frames."); return 2
                frozen=frame; frame_number+=1
            preview=_preview(frozen,frame_number,captured,camera_index,output_dir,session,paused)
            cv2.imshow(window,preview); key=cv2.waitKey(30 if not paused else 100)&0xff
            if key in (ord("q"),ord("Q"),27): break
            if key in (ord("p"),ord("P")): paused=not paused
            elif key==ord(" ") and not paused:
                path=_next_filename(output_dir,session)
                if cv2.imwrite(str(path),frozen,[cv2.IMWRITE_JPEG_QUALITY,95]):
                    captured+=1; print(f"Saved candidate {path}")
                else: print(f"ERROR: could not save {path}")
        print(f"Capture ended. Saved {captured} candidate image(s) to {output_dir}; they are not in a final split.")
        return 0
    except cv2.error as exc: print(f"OpenCV error: {exc}"); return 2
    finally:
        if cap is not None: cap.release()
        try: cv2.destroyAllWindows()
        except cv2.error: pass


def main(argv=None):
    parser=argparse.ArgumentParser(description="Capture manually selected webcam frames into an unreviewed candidate inbox.")
    parser.add_argument("--camera",type=int,default=0); parser.add_argument("--session",default="weapon_session_01")
    parser.add_argument("--output",default="data/weapon/candidates/inbox")
    args=parser.parse_args(argv)
    try: return capture(args.camera,args.session,args.output)
    except ValueError as exc: parser.error(str(exc))


if __name__=="__main__": raise SystemExit(main())
