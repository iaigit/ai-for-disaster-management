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