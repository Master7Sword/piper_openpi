from pathlib import Path
import shutil

def safe_copytree(src: Path, dst: Path):
    if Path(dst).exists():
        print(f"Skip exists: {dst}")
        return
    shutil.copytree(str(src), str(dst))

src_root = Path("/home/tengenx2204/workspace/mozihao/Data/put_item_in_drawer_sup")
dst_root = Path("/home/tengenx2204/workspace/mozihao/Data/put_item_in_drawer_instr")
dst_root.mkdir(parents=True, exist_ok=True)

# for p in sorted(
#     (x for x in src_root.iterdir() if x.is_dir() and x.name.startswith("episode") and x.name.replace("episode", "").isdigit()),
#     key=lambda x: int(x.name.replace("episode", "")),
# ):
#     if p.is_dir() and p.name.startswith("episode"):
#         num = p.name.replace("episode", "")
#         num = int(num)
#         if num <= 69:
#             safe_copytree(p, dst_root / p.name)
#         elif num <= 139:
#             new_num = num + 30
#             new_name = f"episode{new_num}"
#             safe_copytree(p, dst_root / new_name)
#         elif num <= 209:
#             new_num = num + 60
#             new_name = f"episode{new_num}"
#             safe_copytree(p, dst_root / new_name)
#         elif num <= 279:
#             new_num = num + 90
#             new_name = f"episode{new_num}"
#             safe_copytree(p, dst_root / new_name)
#         else:
#             print(f"Not copied: {p.name}")
        

src_root = Path("/home/tengenx2204/workspace/mozihao/Data/put_item_in_drawer")

for p in sorted(
    (x for x in src_root.iterdir() if x.is_dir() and x.name.startswith("episode") and x.name.replace("episode", "").isdigit()),
    key=lambda x: int(x.name.replace("episode", "")),
):
    if p.is_dir() and p.name.startswith("episode"):
        num = p.name.replace("episode", "")
        num = int(num)
        if num <= 68:
            new_num = num + 655
            new_name = f"episode{new_num}"
            safe_copytree(p, dst_root / new_name)
        elif num >= 69 and num <= 117:
            new_num = num - 69 + 724
            new_name = f"episode{new_num}"
            safe_copytree(p, dst_root / new_name)
        elif num >= 118 and num <= 166:
            new_num = num - 118 + 773
            new_name = f"episode{new_num}"
            safe_copytree(p, dst_root / new_name)
        else:
            pass
