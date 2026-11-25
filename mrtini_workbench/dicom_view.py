"""Local, ephemeral monochrome DICOM reference viewer. No patient tags returned."""
import io, base64, hashlib
import numpy as np

MAX_BYTES=64*1024*1024
MAX_PIXELS=8_000_000

def decode_dicom(payload):
    if not payload or len(payload)>MAX_BYTES:raise ValueError('Select a DICOM file smaller than 64 MiB')
    try:import pydicom
    except ImportError as e:raise RuntimeError('Install pydicom to open DICOM references: python -m pip install pydicom') from e
    ds=pydicom.dcmread(io.BytesIO(payload),force=False)
    rows,cols,frames=int(ds.Rows),int(ds.Columns),int(getattr(ds,'NumberOfFrames',1))
    if min(rows,cols,frames)<1 or rows*cols*frames>MAX_PIXELS:raise ValueError('DICOM exceeds the 8 million pixel viewer limit')
    if int(getattr(ds,'SamplesPerPixel',1))!=1 or ds.PhotometricInterpretation not in ('MONOCHROME1','MONOCHROME2'):raise ValueError('Only monochrome DICOM images are supported')
    a=np.asarray(ds.pixel_array).reshape(frames,rows,cols)
    # Enhanced MR can carry frame-specific pixel transforms. Reject those instead of silently applying a global slope.
    for group in [*getattr(ds,'SharedFunctionalGroupsSequence',[]),*getattr(ds,'PerFrameFunctionalGroupsSequence',[])]:
        if hasattr(group,'PixelValueTransformationSequence') or hasattr(group,'RealWorldValueMappingSequence'):raise ValueError('Frame-specific rescale is not yet supported')
    if hasattr(ds,'ModalityLUTSequence'):raise ValueError('Modality LUT is not yet supported')
    a=a.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
    if not np.isfinite(a).all():raise ValueError('Non-finite pixel values')
    def first(name,default):
        x=getattr(ds,name,default)
        try:return float(x)
        except (ValueError,TypeError):return float(x[0])
    lo,hi=np.percentile(a,[1,99]);width=max(1.,first('WindowWidth',hi-lo));center=first('WindowCenter',(lo+hi)/2)
    spacing=[float(v) for v in getattr(ds,'PixelSpacing',[])]
    pos=getattr(ds,'ImagePositionPatient',None);ori=getattr(ds,'ImageOrientationPatient',None)
    position=None
    if pos is not None and ori is not None:
        axis=np.cross(np.asarray(ori[:3],float),np.asarray(ori[3:],float));position=float(np.dot(axis,np.asarray(pos,float)))
    # Opaque grouping token only; no patient identity, accession, UID or dates in response.
    uid=str(getattr(ds,'SeriesInstanceUID',''))
    token=hashlib.sha256(uid.encode()).hexdigest()[:16] if uid else hashlib.sha256(payload).hexdigest()[:16]
    return dict(rows=rows,columns=cols,frames=frames,pixels=base64.b64encode(a.astype('<f4',copy=False).tobytes()).decode(),window_center=center,window_width=width,invert=ds.PhotometricInterpretation=='MONOCHROME1',spacing=spacing,position=position,instance=int(getattr(ds,'InstanceNumber',0) or 0),series=token,modality=str(getattr(ds,'Modality','')),thickness=first('SliceThickness',0),protocol_metadata={key:first(key,0) for key in ('RepetitionTime','EchoTime','FlipAngle','MagneticFieldStrength') if hasattr(ds,key)})


def encode_derived(image,protocol,metadata):
    """Synthetic secondary capture; no fabricated patient orientation or scanner identity."""
    import pydicom
    from pydicom.dataset import FileDataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,SecondaryCaptureImageStorage,generate_uid
    from datetime import datetime
    a=np.asarray(image,float)
    if a.ndim!=2 or not np.isfinite(a).all() or np.min(a)<0:raise ValueError('Expected finite 2D magnitude')
    fm=FileMetaDataset();fm.TransferSyntaxUID=ExplicitVRLittleEndian;fm.MediaStorageSOPClassUID=SecondaryCaptureImageStorage;fm.MediaStorageSOPInstanceUID=generate_uid();fm.ImplementationClassUID=generate_uid()
    ds=FileDataset(None,{},file_meta=fm,preamble=b'\0'*128);ds.SOPClassUID=fm.MediaStorageSOPClassUID;ds.SOPInstanceUID=fm.MediaStorageSOPInstanceUID;ds.StudyInstanceUID=generate_uid();ds.SeriesInstanceUID=generate_uid();ds.PatientName='SIMULATED';ds.PatientID='MRTini';ds.PatientBirthDate='';ds.PatientSex='';ds.AccessionNumber='';ds.ReferringPhysicianName='';ds.StudyID='';ds.SeriesNumber=1;ds.InstanceNumber=1;ds.Modality='OT';ds.ConversionType='WSD';ds.ImageType=['DERIVED','SECONDARY'];ds.SeriesDescription='MRTini simulated magnitude - research only';ds.Manufacturer='MRTini';ds.BurnedInAnnotation='NO';ds.StudyDate=ds.ContentDate=datetime.now().strftime('%Y%m%d');ds.StudyTime=ds.ContentTime=datetime.now().strftime('%H%M%S');ds.Rows,ds.Columns=a.shape;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2';ds.BitsAllocated=ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0;mx=max(float(a.max()),1e-30);ds.PixelData=np.rint(a/mx*65535).astype('<u2').tobytes();ds.RescaleSlope=f'{mx/65535:.9g}';ds.RescaleIntercept='0';ds.RescaleType='US';ds.WindowCenter=f'{mx/2:.9g}';ds.WindowWidth=f'{mx:.9g}';ds.DerivationDescription='Simulation, not acquired MR data; display magnitude scaled to unsigned 16-bit. No patient registration.'
    out=io.BytesIO();pydicom.dcmwrite(out,ds,enforce_file_format=True);return out.getvalue()
