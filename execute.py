import subprocess

models = [
    'faster-rcnn-resnet50-10000',
    'faster-rcnn-resnet50-6000',
    'faster-rcnn-resnet50-5000',
    'faster-rcnn-resnet50-4000',
    'faster-rcnn-resnet50-3000',
    'ssd-mobilenet-v2-10000',
    'ssd-mobilenet-v2-6000',
    'ssd-mobilenet-v2-5000',
    'ssd-mobilenet-v2-4000',
    'ssd-mobilenet-v2-3000',
    'ssd-resnet50-fpn-10000',
    'ssd-resnet50-fpn-5000',
]

for model in models:
    print('start executing model: ', model)
    commmand_to_execute = 'sudo python3 detect_objects.py --threshold 0.3 --model_path models/' + model + ' --path_to_labelmap models/shrimp-seed_label_map.pbtxt --images_dir data/test16 --output_directory data/output/' + model + ' --save_output'
    subprocess.call(commmand_to_execute, shell=True)
    print(model + ' done!')