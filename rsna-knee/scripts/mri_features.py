"""Identical three-plane MRI preprocessing and ImageNet-only encoder for both arms."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut
from PIL import Image

LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
PLANES=['Sagittal','Coronal','Axial']

def find_input_root(relative_file,base=Path('/kaggle/input')):
    """Bounded mount discovery; never recursively enumerate millions of DICOMs."""
    level=[base]
    for _ in range(5):
        for path in level:
            if (path/relative_file).is_file():return path
        level=[child for path in level for child in path.iterdir()
               if child.is_dir() and child.name not in ['train_series','test_series','cache','connectome']]
    raise FileNotFoundError('Input mount missing: '+str(relative_file))

def study_images(data, split, uid, series, slices=16, size=224):
    images=[]; planes=[]; source=[]
    for plane_no,plane in enumerate(PLANES):
        rows=series[(series.StudyInstanceUID==uid)&(series.Anatomical_Plane==plane)].copy()
        rows['priority']=2*pd.to_numeric(rows.Fluid_Sensitive)+pd.to_numeric(rows.Fat_Suppression)
        rows=rows.sort_values(['priority','SeriesInstanceUID'],ascending=[False,True])
        if rows.empty: continue
        row=rows.iloc[0]; folder=data/(split+'_series')/uid/row.SeriesInstanceUID
        ordered=[]
        for file in folder.glob('*.dcm'):
            ds=pydicom.dcmread(file,stop_before_pixels=True)
            orientation=np.array(ds.get('ImageOrientationPatient',[1,0,0,0,1,0]),float)
            position=np.array(ds.get('ImagePositionPatient',[0,0,0]),float)
            order=float(np.dot(np.cross(orientation[:3],orientation[3:]),position))
            ordered.append((order,int(ds.get('InstanceNumber',0)),file))
        ordered.sort()
        if not ordered: raise RuntimeError('No DICOM files in '+str(folder))
        for index in np.unique(np.linspace(0,len(ordered)-1,min(slices,len(ordered))).round().astype(int)):
            file=ordered[index][2];ds=pydicom.dcmread(file)
            pixels=apply_modality_lut(ds.pixel_array,ds).astype(np.float32)
            if pixels.ndim!=2: raise ValueError('Expected single-frame grayscale slice')
            lo,hi=np.percentile(pixels,[1,99]); pixels=np.clip((pixels-lo)/max(hi-lo,1e-6),0,1)
            if ds.get('PhotometricInterpretation')=='MONOCHROME1':pixels=1-pixels
            pixels=np.asarray(Image.fromarray((pixels*255).astype(np.uint8)).resize((size,size),Image.Resampling.BILINEAR))
            images.append(pixels);planes.append(plane_no)
            source.append(str(file.relative_to(data)))
    if not images:raise RuntimeError('No usable series: '+uid)
    return np.stack(images),np.array(planes),source

def prepare_cache(data,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(data/'train.csv',dtype={'StudyInstanceUID':str})
    train=train[train[LABELS].notna().all(axis=1)].copy()
    assert len(train)==58,'Source changed: review sample count before running'
    train[['StudyInstanceUID']+LABELS].to_csv(out/'labels.csv',index=False)
    records=[];times=[]
    for split,frame in [('train',train),('test',pd.read_csv(data/'test.csv',dtype=str))]:
        series=pd.read_csv(data/(split+'_series.csv'),dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
        for uid in frame.StudyInstanceUID:
            t=time.perf_counter();images,planes,source=study_images(data,split,uid,series)
            np.savez_compressed(out/(uid+'.npz'),images=images,planes=planes)
            records.append({'uid':uid,'split':split,'slices':len(images),'source':source})
            times.append(time.perf_counter()-t)
            print('prepared',len(records),split,len(images),round(times[-1],2),flush=True)
    (out/'image_manifest.json').write_text(json.dumps(records,indent=2))
    (out/'preprocess_resources.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'seconds_per_study':times,'image_size':224,'max_slices_per_plane':16,'selection':'highest 2*fluid_sensitive+fat_suppression, UID tie break; geometric slice order'},indent=2))

def load_encoder(weights,device):
    import torch,torchvision
    model=torchvision.models.resnet50(weights=None)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.fc=torch.nn.Identity();return model.to(device).eval()

def encode_images(model,images,planes,device):
    import torch
    features=[]
    with torch.inference_mode():
        for i in range(0,len(images),16):
            x=torch.from_numpy(images[i:i+16].copy()).to(device).float()[:,None].repeat(1,3,1,1)/255
            mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
            # Same full-image resize in training and inference, no stochastic augmentation.
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                f=model((x-mean)/std)
            features.append(f.float().cpu().numpy())
    f=np.concatenate(features)
    result=[]
    for plane in range(3):
        selected=f[planes==plane]
        result.extend([selected.mean(0),selected.max(0)] if len(selected) else [np.zeros(2048),np.zeros(2048)])
    return np.concatenate(result).astype(np.float32)

def extract_cache(cache,weights,out):
    import torch,psutil,platform
    started=time.perf_counter();device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    model=load_encoder(weights,device);features=[];ids=[];times=[];peak_rss=0
    records=json.loads((cache/'image_manifest.json').read_text())
    for record in records:
        t=time.perf_counter();z=np.load(cache/(record['uid']+'.npz'))
        features.append(encode_images(model,z['images'],z['planes'],device));ids.append(record['uid'])
        if device=='cuda':torch.cuda.synchronize()
        times.append(time.perf_counter()-t);peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
        print('encoded',len(ids),round(times[-1],3),flush=True)
        if len(ids)%10==0:np.savez_compressed(out/'features_partial.npz',x=np.stack(features),ids=np.array(ids))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'features.npz',x=np.stack(features),ids=np.array(ids))
    (out/'encoder_resources.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'study_seconds':times,'rss_sampled_peak_bytes':peak_rss,'vram_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'vram_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,'device':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),'torch':torch.__version__,'feature_dimension':12288,'weights':'torchvision ResNet50 IMAGENET1K_V2 resnet50-11ad3fa6.pth'},indent=2))
