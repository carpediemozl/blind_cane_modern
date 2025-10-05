from depthai_utils import V5blindPipeline

modelPath = r'models/path_segmentation.blob'
label = ['blind_path', 'crosswalk']
crosswalkPipeline = V5blindPipeline(modelPath, label)

try:
    crosswalkPipeline.run()
except:
    raise
