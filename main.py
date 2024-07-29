import os
import csv
import torch
from tqdm import tqdm

from models.clap_encoder import CLAP_Encoder
from dataset import AudioTextDataset4CLAP, get_dataloader

def CLAPScore_Calculator(
    sampling_rate, text_queries, audio_dir,
    save_dir="./CLAPScore_results",
    csv_filename="CLAPScore_scores",
    pretrained_checkpoint="checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt"
):
    dataset = AudioTextDataset4CLAP(sampling_rate=sampling_rate, text_queries=text_queries, audio_dir=audio_dir)
    dataloader = get_dataloader(
        dataset,
        batch_size=32, # should > 1
        num_workers=4,
        shuffle=False,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = CLAP_Encoder(
        device=device,
        pretrained_path=pretrained_checkpoint,
    ).eval()

    scores = torch.tensor([]).to(device)
    csv_objs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            waveform = batch['waveform'].to(device)
            caption = batch['caption']
            filename = batch['filename']

            conditions = model.get_query_embed(
                modality='text',
                text=caption,
                device=device
            )
            audios = model.get_query_embed(
                modality='audio',
                audio=waveform.squeeze().to(device),
                device=device
            )
            # batch_scores = F.cosine_similarity(conditions, audios, dim=1)
            batch_scores = (conditions * audios).sum(-1)
            # print(batch_scores.shape)
            scores = torch.cat((scores, batch_scores), 0)
            scores = scores.squeeze()
            for i, (wav_name, caption) in enumerate(zip(filename, caption)):
                csv_objs.append({
                    "filename": wav_name,
                    "caption": caption,
                    "CLAP_score": batch_scores[i].item(),
                })

    print(f"Average CLAPScore: {scores.mean()}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 写入 CSV 文件
    save_file = f"{save_dir}/{csv_filename}.csv"
    with open(save_file, mode='w', newline='') as csv_file:
        # 获取表头
        keys = csv_objs[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=keys)

        # 写入表头
        writer.writeheader()

        # 写入数据行
        for item in csv_objs:
            writer.writerow(item)

    print(f"CLAPScore values have been written to {save_file}")

    return scores

if __name__ == "__main__":
    sampling_rate = 16000
    text_queries = "text_queries.csv"
    audio_dir = "audio_dir"

    all_CLAPScores = CLAPScore_Calculator(sampling_rate=sampling_rate, text_queries=text_queries, audio_dir=audio_dir)
