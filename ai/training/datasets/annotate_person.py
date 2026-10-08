"""Local Tk annotation UI for person boxes; writes project-format JSONL directly."""
import argparse
import json
import math
from pathlib import Path
from PIL import Image

IMAGE_SUFFIXES={".jpg",".jpeg",".png"}


def person_record(relative_image, boxes, width, height):
    """Build one valid person manifest record from original-image pixel boxes."""
    objects=[]
    for box in boxes:
        if len(box)!=4: raise ValueError("Each box must be [x1,y1,x2,y2]")
        if any(isinstance(value,bool) for value in box): raise ValueError("Boolean box coordinates are invalid")
        try: coords=list(map(float,box))
        except (TypeError,ValueError) as exc: raise ValueError("Box coordinates must be numeric") from exc
        x1,y1,x2,y2=coords
        if not all(math.isfinite(value) for value in coords) or not (0<=x1<x2<=width and 0<=y1<y2<=height):
            raise ValueError(f"Person box {box!r} is outside the {width}x{height} image")
        pixel_box=[int(value) if value.is_integer() else value for value in coords]
        objects.append({"class":0,"bbox":pixel_box})
    return {"image":Path(relative_image).as_posix(),"objects":objects}


def preserve_record_metadata(record, previous):
    """Retain source/license provenance when a reviewed box record is updated."""
    metadata = {key: value for key, value in (previous or {}).items()
                if key not in {"image", "objects"}}
    return {**metadata, **record, "reviewed": True}


def _launch(data_root, split):
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise RuntimeError("This Python has no Tk support. Install the OS tkinter package for this interpreter, or use a local COCO-exporting annotation tool and ai.training.datasets.convert.") from exc
    from PIL import ImageTk

    root=Path(data_root).resolve(); image_dir=root/"images"; output=root/f"{split}.jsonl"
    if not image_dir.is_dir(): raise FileNotFoundError(f"Image folder not found: {image_dir}")
    files=sorted((p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),key=lambda p:p.as_posix().lower())
    if not files: raise FileNotFoundError(f"No JPG/PNG images found under {image_dir}")
    records={}
    if output.exists():
        for line_no,line in enumerate(output.read_text(encoding="utf-8").splitlines(),1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get("image"),str) or not isinstance(row.get("objects"),list):
                raise ValueError(f"Malformed annotation at {output}:{line_no}")
            records[row["image"]]=row

    window=tk.Tk(); window.title(f"Person annotation — {root.name} / {split}")
    index=0; boxes=[]; image_size=(0,0); scale=1.0; photo=None; drag=None; rectangle=None
    canvas=tk.Canvas(window,width=1000,height=650,bg="#202020",highlightthickness=0); canvas.pack(padx=8,pady=8)
    status=tk.StringVar(); reviewed=tk.BooleanVar(value=False)
    info=tk.Label(window,textvariable=status,anchor="w"); info.pack(fill="x",padx=8)
    controls=tk.Frame(window); controls.pack(fill="x",padx=8,pady=8)
    tk.Checkbutton(controls,text="Reviewed: this image contains no people",variable=reviewed).pack(side="left")

    def relpath(path): return path.relative_to(root).as_posix()

    def render_boxes():
        nonlocal rectangle
        canvas.delete("box")
        for n,(x1,y1,x2,y2) in enumerate(boxes,1):
            canvas.create_rectangle(x1*scale,y1*scale,x2*scale,y2*scale,outline="#ff3333",width=2,tags="box")
            canvas.create_text(x1*scale+4,y1*scale+4,text=f"Person {n}",fill="white",anchor="nw",tags="box")

    def show_image():
        nonlocal boxes,image_size,scale,photo
        path=files[index]; rel=relpath(path)
        with Image.open(path) as im: image=im.convert("RGB")
        image_size=image.size
        scale=min(1000/image.width,650/image.height,1.0)
        shown=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))))
        photo=ImageTk.PhotoImage(shown)
        canvas.configure(width=shown.width,height=shown.height); canvas.delete("all")
        canvas.create_image(0,0,image=photo,anchor="nw")
        row=records.get(rel,{"objects":[]})
        boxes=[list(map(int,obj["bbox"])) for obj in row.get("objects",[])]
        reviewed.set(not boxes and row.get("reviewed") is True)
        status.set(f"{index+1}/{len(files)}  {rel}  ({image.width}x{image.height})  — draw tight boxes around each visible person")
        render_boxes()

    def point(event):
        return min(max(event.x,0),image_size[0]*scale),min(max(event.y,0),image_size[1]*scale)

    def press(event):
        nonlocal drag,rectangle
        drag=point(event); reviewed.set(False)
        if rectangle: canvas.delete(rectangle)
        x,y=drag; rectangle=canvas.create_rectangle(x,y,x,y,outline="#ffff00",width=2)

    def move(event):
        if drag and rectangle:
            x,y=point(event); canvas.coords(rectangle,drag[0],drag[1],x,y)

    def release(event):
        nonlocal drag,rectangle,boxes
        if not drag:return
        end=point(event); x1=round(min(drag[0],end[0])/scale); y1=round(min(drag[1],end[1])/scale)
        x2=round(max(drag[0],end[0])/scale); y2=round(max(drag[1],end[1])/scale)
        drag=None
        if rectangle: canvas.delete(rectangle); rectangle=None
        if x2-x1<2 or y2-y1<2: return
        boxes.append([max(0,x1),max(0,y1),min(image_size[0],x2),min(image_size[1],y2)])
        reviewed.set(False); render_boxes()

    def save_current():
        rel=relpath(files[index])
        if not boxes and not reviewed.get():
            messagebox.showwarning("Review empty image","Draw person boxes, or check the reviewed-empty box before saving.",parent=window)
            return False
        try: records[rel]=preserve_record_metadata(
            person_record(rel,boxes,*image_size), records.get(rel))
        except ValueError as exc:
            messagebox.showerror("Invalid box",str(exc),parent=window); return False
        tmp=output.with_suffix(output.suffix+".tmp")
        try:
            with tmp.open("w",encoding="utf-8") as stream:
                for row in records.values(): stream.write(json.dumps(row,separators=(",",":"))+"\n")
            tmp.replace(output)
        except OSError as exc:
            messagebox.showerror("Save failed",str(exc),parent=window); return False
        return True

    def next_image():
        nonlocal index
        if not save_current(): return
        if index<len(files)-1: index+=1; show_image()
        else: messagebox.showinfo("Finished","Saved the final image in this folder.",parent=window)

    def previous_image():
        nonlocal index
        if not save_current(): return
        if index>0: index-=1; show_image()

    def clear_boxes():
        nonlocal boxes
        boxes=[]; reviewed.set(False); render_boxes()

    tk.Button(controls,text="Previous",command=previous_image).pack(side="left",padx=4)
    tk.Button(controls,text="Save",command=save_current).pack(side="left",padx=4)
    tk.Button(controls,text="Next",command=next_image).pack(side="left",padx=4)
    tk.Button(controls,text="Clear boxes",command=clear_boxes).pack(side="left",padx=4)
    canvas.bind("<ButtonPress-1>",press); canvas.bind("<B1-Motion>",move); canvas.bind("<ButtonRelease-1>",release)
    show_image(); window.mainloop()


def main(argv=None):
    parser=argparse.ArgumentParser(description="Annotate person boxes locally and write train/val/test JSONL.")
    parser.add_argument("--data",default="data/person",help="Person dataset root")
    parser.add_argument("--split",choices=("train","val","test"),default="train")
    args=parser.parse_args(argv); _launch(args.data,args.split); return 0


if __name__=="__main__": raise SystemExit(main())
