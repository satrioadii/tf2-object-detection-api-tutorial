import subprocess

models = [
    'faster-rcnn-resnet50-8000',
    'faster-rcnn-resnet50-6000',
    'faster-rcnn-resnet50-5000',
    'faster-rcnn-resnet50-4000',
    # 'ssd-mobilenet-v2-8000',
    # 'ssd-mobilenet-v2-6000',
    # 'ssd-mobilenet-v2-5000',
    # 'ssd-mobilenet-v2-4000',
]

threshold_setup = [0.3, 0.2]
test_images_folders = ['test17', 'test33']

for threshold in threshold_setup:
    
    # Generate string for threshold output folder
    threshold_str = str(threshold)
    threshold_str = threshold_str.replace('.', '_')

    for folder in test_images_folders:

        # Generate string for output folder
        folder_subname = folder.replace('text', '')

        for model in models:
            print('start executing [folder: ' + folder + '] ' + ' [threshold: ' + str(threshold) + ']: ' + model )
            
            # Generate output directory
            output_directory = 'output_' + folder_subname + '_' + threshold_str

            # Generate command to execute [on terminal]
            commmand_to_execute = 'sudo python3 detect_objects.py --threshold ' + str(threshold) +' --model_path models/' + model + ' --path_to_labelmap models/shrimp-seed_label_map.pbtxt --images_dir data/' + folder +' --output_directory data/' + output_directory + '/' + model + ' --save_output'
            subprocess.call(commmand_to_execute, shell=True)

            print(model +  ' [folder: ' + folder + '] ' + ' [threshold: ' + str(threshold) + ']: DONE!')