"""Independent analytic 35 cm cube renderer; requires NumPy and OpenCV Python.

Creates synthetic projection checks, not physical camera accuracy measurements.
Run from any directory: python tests/render_motion_fixtures.py
"""
from pathlib import Path
import json
import cv2
import numpy as np

root=Path(__file__).resolve().parent/'fixtures/motion'
root.mkdir(exist_ok=True)
half=.175
cube=np.array([[x,y,z] for x in [-half,half] for y in [-half,half] for z in [-half,half]],np.float64)
width,height,focal=640,480,500.
intrinsics=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
background=cv2.imread(str(root.parent/'negative.jpg'))
faces=[([0,4,6,2],[0,0,-1]),([1,3,7,5],[0,0,1]),([0,2,3,1],[-1,0,0]),([4,5,7,6],[1,0,0]),([0,1,5,4],[0,-1,0]),([2,6,7,3],[0,1,0])]
cases=[((15,25,0),3),((-20,55,15),3),((30,70,-25),3),((0,45,40),3),((0,0,0),3),((10,5,0),1.5),((-15,75,25),4),((40,35,-20),2),((20,35,-15),3),((-20,65,25),2.5)]
scenes=[]
for index,(angles,desired) in enumerate(cases):
    rx,ry,rz=np.deg2rad(angles)
    x=np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    y=np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    z=np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    rotation=z@y@x
    low,high=.5,10
    for _ in range(40):
        translation=np.array([0,0,(low+high)/2],np.float64)
        # Camera in cube coordinates; distance to the closest point on the box.
        distance=float(np.linalg.norm(np.maximum(np.abs(-rotation.T@translation)-half,0)))
        if distance<desired:low=translation[2]
        else:high=translation[2]
    vertices=cv2.projectPoints(cube,cv2.Rodrigues(rotation)[0],translation,intrinsics,None)[0].reshape(-1,2)
    image=cv2.resize(background,(width*4,height*4))
    for indices,normal in sorted(faces,key=lambda face:np.mean((cube[face[0]]@rotation.T+translation)[:,2]),reverse=True):
        centre=rotation@cube[indices].mean(0)+translation
        normal=rotation@np.array(normal)
        if normal@centre>=0:continue
        intensity=.55+.45*max(0,normal@np.array([-.3,-.6,-.74]))
        colour=tuple(int(v*intensity) for v in ((210,90,35) if index<8 else (35,60,220)))
        points=np.rint(vertices[indices]*4).astype(np.int32)
        cv2.fillConvexPoly(image,points,colour,cv2.LINE_AA)
        cv2.polylines(image,[points],True,tuple(int(v*.8) for v in colour),2,cv2.LINE_AA)
    image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
    name=f'motion-{index}.jpg'
    cv2.imwrite(str(root/name),image,[cv2.IMWRITE_JPEG_QUALITY,97])
    scenes.append({'name':name,'angles':angles,'truth_metres':distance,'focal':focal,'classId':0 if index<8 else 1,
                   'points':cv2.convexHull(vertices.astype(np.float32)).reshape(-1,2).tolist(),
                   'box':[float(vertices[:,0].min()),float(vertices[:,1].min()),float(vertices[:,0].max()),float(vertices[:,1].max())]})
(root/'scenes.json').write_text(json.dumps(scenes,indent=2)+'\n')
print(f'Rendered {len(scenes)} synthetic cube viewpoints in {root}')
