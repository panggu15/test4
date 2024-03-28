import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
# import warnings
# warnings.filterwarnings(action='ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':2, #Your Epochs,
    'LR':5e-5, #Your Learning Rate,
    'BATCH_SIZE':4, #Your Batch Size,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration
from transformers import TrainingArguments, Trainer

model_id = "Salesforce/blip-image-captioning-base"

processor = AutoProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
model.init_weights()

import datasets
from datasets import Dataset

train_data = pd.read_csv('/kaggle/working/data.csv')

# train_data.loc[12263, 'comments'] = train_data.loc[12263, 'comments'].replace('. ,', ',')

# def fix_sentence(sentence):
#     sentence = re.sub(r'^[^a-zA-Z0-9]+', '', sentence)
#     sentence = re.sub(r'([A-Za-z0-9])([.,;!?])|([A-Za-z0-9]) ([.,;!?])', r'\1\3\2\4 ', sentence)
#     sentence = re.sub(r'\s+', ' ', sentence)
#     sentence = sentence.replace('. .', '.').replace(', ,', ',').replace('! !', '!')
#     sentence = sentence.replace('. .', '.').replace(', ,', ',').replace('! !', '!')
#     return sentence.strip()

# train_data['comments'] = train_data['comments'].map(fix_sentence)

data_dict = {
    'image': train_data['url'].tolist(),
    'text': train_data['text'].tolist(),
}

train_dataset = Dataset.from_dict(data_dict)
train_dataset = train_dataset.cast_column('image', datasets.Image())

def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    # for i, cap in enumerate(captions):
        # p = np.random.random()
        # if p < 0.3:
        #     split_list = [part for part in cap.split('.') if part]
        #     random.shuffle(split_list)
        #     shuffled_string = '.'.join(split_list)
        #     captions[i] = shuffled_string.strip()
        # elif p < 0.6:
        #     split_list = [part for part in cap.split('.') if part]
        #     if len(split_list) > 1:
        #         index_to_remove = random.randint(0, len(split_list) - 1)
        #         del split_list[index_to_remove]
        #     shuffled_string = '.'.join(split_list)
        #     captions[i] = shuffled_string.strip()
            
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

train_dataset.set_transform(transforms)
valid_dataset = train_dataset.select(list(range(1000)))


training_args = TrainingArguments(
    output_dir = "./",
    logging_dir = './logs',
    logging_steps = 50,
    learning_rate = CFG['LR'],
    num_train_epochs=CFG['EPOCHS'],
    per_device_train_batch_size=CFG['BATCH_SIZE'],
    per_device_eval_batch_size=CFG['BATCH_SIZE'],
    gradient_accumulation_steps=4,
    save_total_limit=2,
    remove_unused_columns=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    seed=CFG['SEED'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

# class CaptionDataset(torch.utils.data.Dataset):
#     def __init__(self, dataframe, processor, mode='train'):
#         self.dataframe = dataframe
#         self.processor = processor
#         self.mode = mode
        
#     def __len__(self):
#         return len(self.dataframe)
    
#     def __getitem__(self, idx):
#         img_path = self.dataframe.iloc[idx]['img_path']
#         img = Image.open(img_path).convert('RGB')
        
#         if self.mode == 'train':
#             text = self.dataframe.iloc[idx]['comments']
#             encoding = self.processor(images=img, text=text, padding="max_length", return_tensors="pt")
#         else:
#             encoding = self.processor(images=img, padding="max_length", return_tensors="pt")
            
#         encoding = {k:v.squeeze() for k,v in encoding.items()}
#         return encoding


# test_data = pd.read_csv('test.csv')
# test_dataset = CaptionDataset(test_data, processor, mode='test')
# test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# model.eval()
# predicted_mos_list = [0]
# predicted_comments_list = []

# # 추론 과정
# with torch.no_grad():
#     for batch in tqdm(test_loader):
#         pixel_values = batch.pop("pixel_values").to(device)
#         generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
#         caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
#         predicted_comments_list.extend(caption)

# # 결과 저장
# result_df = pd.DataFrame({
#     'img_name': test_data['img_name'],
#     'mos': predicted_mos_list*len(test_data['img_name']),
#     'comments': predicted_comments_list  # 캡션 부분은 위에서 생성한 것을 사용
# })

# # 예측 결과에 NaN이 있다면, 제출 시 오류가 발생하므로 후처리 진행 (sample_submission.csv과 동일하게)
# result_df['comments'] = result_df['comments'].fillna('Nice Image.')
# result_df.to_csv('../wav.csv', index=False)
