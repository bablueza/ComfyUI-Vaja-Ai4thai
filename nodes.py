import requests
import logging
import torch
import librosa
logger = logging.getLogger(__name__)

class Ai4thaiVaja:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apikey": ("STRING", {"default": ""}),
                "input_text": ("STRING", {"default": "ทดสอบการสังเคราะห์เสียงพูด"}),
                "speaker": ("INT", {"default": 1}),
            },
        }

    # RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("text_out",)
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "ai4thai"
    FUNCTION = "sythesis"

    def sythesis(self, apikey, input_text, speaker):
        url = 'https://api.aiforthai.in.th/vaja9/synth_audiovisual'
        headers = {'Apikey':apikey,'Content-Type' : 'application/json'}
        data = {'input_text':input_text,'speaker': speaker, 'phrase_break':0, 'audiovisual':0}
        response = requests.post(url, json=data, headers=headers)
        logger.info(response.json())

        # ดาวน์โหลดไฟล์เสียง
        resp = requests.get(response.json()['wav_url'],headers={'Apikey':apikey})
        if resp.status_code == 200:
            with open('output/vaja_sync_out.wav', 'wb') as a:
                a.write(resp.content)
        
        data , sample_rate = librosa.load('output/vaja_sync_out.wav')
        audio_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)