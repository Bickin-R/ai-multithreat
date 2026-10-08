"""Manual OpenCV box annotation; only explicit A/N review writes final JSONL."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
from .prepare_weapon_dataset import DATA_ROOT, _read_index, _write_index, sha256

CLASSES=("knife","gun","arivaal","baseball_bat","other_weapon")


class WeaponAnnotator:
    def __init__(self, root, split):
        self.root=Path(root); self.split=split
        self.index_path=self.root/"candidates"/"index.jsonl"
        self.rows=_read_index(self.index_path)
        self.pending=[row for row in self.rows if row.get("status")=="pending"]
        if not self.pending: raise ValueError("No pending candidates. Import local images with prepare_weapon_dataset first.")
        self.position=0; self.boxes=[]; self.current_class=0; self.selected=None
        self.drag=None; self.drag_mode=None; self.preview=None; self.image=None; self.window="Weapon annotation review"

    def _load(self):
        row=self.pending[self.position]; path=self.root/row["candidate"]
        self.image=cv2.imread(str(path))
        if self.image is None: raise ValueError(f"Cannot read candidate image: {path}")
        self.boxes=[]; self.selected=None

    @staticmethod
    def _inside(point, box):
        x,y=point; x1,y1,x2,y2=box["bbox"]
        return x1<=x<=x2 and y1<=y<=y2

    def _mouse(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.drag=(x,y); self.drag_mode="draw"; self.selected=None
            for index in range(len(self.boxes)-1,-1,-1):
                box=self.boxes[index]["bbox"]
                if abs(x-box[2])<=12 and abs(y-box[3])<=12:
                    self.selected=index; self.drag_mode="resize"; break
                if self._inside((x,y),self.boxes[index]):
                    self.selected=index; self.drag_mode="move"; break
        elif event==cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.preview=(x,y)
        elif event==cv2.EVENT_LBUTTONUP and self.drag is not None:
            x0,y0=self.drag; x1,y1=x,y
            if self.drag_mode=="draw" and abs(x1-x0)>=3 and abs(y1-y0)>=3:
                self.boxes.append({"class":self.current_class,"bbox":[min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)]})
                self.selected=len(self.boxes)-1
            elif self.drag_mode=="move" and self.selected is not None:
                box=self.boxes[self.selected]["bbox"]; dx=x1-x0; dy=y1-y0
                width,height=self.image.shape[1],self.image.shape[0]
                bw,bh=box[2]-box[0],box[3]-box[1]
                nx=max(0,min(width-bw,box[0]+dx)); ny=max(0,min(height-bh,box[1]+dy))
                self.boxes[self.selected]["bbox"]=[nx,ny,nx+bw,ny+bh]
            elif self.drag_mode=="resize" and self.selected is not None:
                box=self.boxes[self.selected]["bbox"]
                box[2]=max(box[0]+2,min(self.image.shape[1],x)); box[3]=max(box[1]+2,min(self.image.shape[0],y))
            self.drag=None; self.drag_mode=None; self.preview=None

    def _render(self):
        frame=self.image.copy()
        for index,item in enumerate(self.boxes):
            x1,y1,x2,y2=item["bbox"]; color=(0,220,0) if index!=self.selected else (0,180,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,CLASSES[item["class"]],(x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
            cv2.circle(frame,(x2,y2),5,color,-1)
        if self.drag is not None and self.preview is not None and self.drag_mode=="draw":
            cv2.rectangle(frame,self.drag,self.preview,(255,200,0),1)
        row=self.pending[self.position]
        lines=[f"{self.position+1}/{len(self.pending)}  {row['source_name']}  group={row.get('source_group','?')}",
               f"Draw class {self.current_class}: {CLASSES[self.current_class]} | boxes={len(self.boxes)} | selected={self.selected}",
               "Drag blank area: draw | drag box: move | drag orange corner: resize",
               "0-4 class  X delete selected  A accept  N reviewed negative  S skip  Q quit",
               "Review only objects/props you are authorized to photograph; use safe training props."]
        y=25
        for line in lines:
            cv2.putText(frame,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(frame,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA); y+=23
        return frame

    def _promote(self, negative=False):
        row=self.pending[self.position]
        if not negative and not self.boxes:
            print("No boxes drawn. Press N only after explicitly reviewing an empty image."); return
        image_rel=row["candidate"].replace("candidates/images/","images/")
        for split_name in ("train","val","test"):
            manifest=self.root/f"{split_name}.jsonl"
            if manifest.is_file():
                for line in manifest.read_text(encoding="utf-8").splitlines():
                    try: existing=json.loads(line)
                    except json.JSONDecodeError: continue
                    if isinstance(existing,dict) and existing.get("image")==image_rel:
                        raise ValueError(f"Image already exists in {split_name}.jsonl; inspect existing annotation before continuing")
        source=self.root/row["candidate"]; destination=self.root/image_rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and sha256(destination)!=row["sha256"]:
            destination=self.root/"images"/(Path(image_rel).stem+"_"+row["sha256"][:8]+Path(image_rel).suffix)
            image_rel=destination.relative_to(self.root).as_posix()
        if not destination.exists(): shutil.copy2(source,destination)
        objects=[] if negative else [{"class":item["class"],"bbox":item["bbox"]} for item in self.boxes]
        manifest=self.root/f"{self.split}.jsonl"
        record={"image":image_rel,"objects":objects,"reviewed":True,
                "source_group":row.get("source_group",row.get("source_name","unknown"))}
        with manifest.open("a",encoding="utf-8") as stream: stream.write(json.dumps(record,separators=(",",":"))+"\n")
        row["status"]="reviewed_negative" if negative else "reviewed_positive"
        row["split"]=self.split
        _write_index(self.index_path,self.rows)
        self.next()

    def next(self):
        self.position+=1
        if self.position<len(self.pending): self._load()

    def run(self):
        self._load(); cv2.namedWindow(self.window,cv2.WINDOW_NORMAL); cv2.setMouseCallback(self.window,self._mouse)
        try:
            while self.position<len(self.pending):
                cv2.imshow(self.window,self._render()); key=cv2.waitKey(30)&0xff
                if key in range(ord("0"),ord("5")):
                    self.current_class=key-ord("0")
                    if self.selected is not None: self.boxes[self.selected]["class"]=self.current_class
                elif key in (ord("x"),ord("X"),8,127) and self.selected is not None:
                    self.boxes.pop(self.selected); self.selected=None
                elif key in (ord("a"),ord("A")): self._promote(False)
                elif key in (ord("n"),ord("N")):
                    self.boxes=[]; self._promote(True)
                elif key in (ord("s"),ord("S")): self.next()
                elif key in (ord("q"),ord("Q"),27): break
        finally: cv2.destroyAllWindows()
        print(f"Review session ended. Remaining pending: {max(0,len(self.pending)-self.position)}")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Manually annotate candidates. A/N explicitly approve and promote to a final split.")
    parser.add_argument("--data",default=str(DATA_ROOT)); parser.add_argument("--split",choices=("train","val","test"),required=True,
                        help="Choose one split and keep each source/capture session wholly in that split")
    args=parser.parse_args(argv)
    try: WeaponAnnotator(args.data,args.split).run()
    except (ValueError,cv2.error) as exc: parser.error(str(exc))
    return 0


if __name__=="__main__": raise SystemExit(main())
