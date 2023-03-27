import os
from dotenv import load_dotenv

load_dotenv()

epochs = int(os.getenv('EPOCHS'))
model_type = os.getenv('MODEL_TYPE')
criterion_type = os.getenv('CRITERION_TYPE')
image_size = int(os.getenv('IMAGE_SIZE'))
mean_normalize = [float(x) for x in os.getenv('MEAN_NORMALIZE').split(',')]
std_normalize = [float(x) for x in os.getenv('STD_NORMALIZE').split(',')]
data_name = os.getenv('DATA_NAME')
trainer_root_dir = os.getenv('TRAINER_ROOT_DIR')
checkpoint_dir = os.path.join(trainer_root_dir, 'checkpoints')
train_annotations_file = os.getenv('TRAIN_ANNOTATIONS_FILE')
val_annotations_file = os.getenv('VAL_ANNOTATIONS_FILE')
test_annotations_file = os.getenv('TEST_ANNOTATIONS_FILE')
batch_size = int(os.getenv('BATCH_SIZE'))
num_workers = int(os.getenv('NUM_WORKERS'))
pin_memory = os.getenv('PIN_MEMORY') == 'True'
learning_rate = float(os.getenv('LEARNING_RATE'))
weight_decay = float(os.getenv('WEIGHT_DECAY'))
mode_segmentation = os.getenv('MODE_SEGMENTATION')
num_classes = int(os.getenv('NUM_CLASSES'))
mask_threshold = float(os.getenv('MASK_THRESHOLD'))
fp16 = os.getenv('FP16') == 'True'
gradient_clip = float(os.getenv('GRADIENT_CLIP'))