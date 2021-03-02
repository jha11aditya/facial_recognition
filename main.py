import os
import sys


op_path = "dataset"

if len(sys.argv) < 2:
    print("no input raw data folder")
    exit(0)

rdat = sys.argv[1]
print(rdat, [name for name in os.listdir(rdat) if os.path.isdir(os.path.join(rdat, name))])

subj_names = [name for name in os.listdir(rdat) if os.path.isdir(os.path.join(rdat, name)) ]

os.system("mkdir -p dataset")

for d in subj_names:
    inpath = rdat + "/" + d
    outpath = op_path + "/" + d
    os.system("mkdir -p " + outpath)
    img_files = [name for name in os.listdir(inpath) if  not os.path.isdir(os.path.join(inpath, name)) ]
    # print(img_files)
    for img in img_files:
        inimg = inpath + "/" + img
        # print(inimg)
        # os.system("convert -verbose -coalesce " + inimg + " " + inimg + ".png")
        # os.system("rm  "+ inimg)
        os.system("python3 face_cutter.py " + inimg + " " + outpath )
        
    inpath2 = outpath
    outpath2 = outpath
    img_files2 = [name for name in os.listdir(inpath2) if  not os.path.isdir(os.path.join(inpath2, name)) ]
    print(inpath2,img_files2)
    for img2 in img_files2:
        inimg2 = inpath2 + "/" + img2
        
        os.system("python3 hogger.py " + inimg2 + " " + outpath2 )
        os.system("rm " + inimg2)
    # break
        
