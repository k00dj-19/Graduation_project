import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math 
import json
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def make_caption(image_path, model, tokenizer, generation_config, prompt):
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    response = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None)
    return response

def extract_vids_from_jsonl(file_path):
    vid_list = []
    
    # 파일을 열고 각 줄을 처리
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)  # json 형식으로 변환
            vid_list.append(data['vid'])  # vid 값 추출 후 리스트에 추가
    vid_list = list(set(vid_list))
    return vid_list

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_captions():
    frame_dir = '/data/Shared_Data/qvhighlights/video_frames_real/'
    #train_vid_list = extract_vids_from_jsonl('./highlight_train_release.jsonl')
    # train_vid_split_1 = extract_vids_from_jsonl('./highlight_train_split_1.jsonl')
    # train_vid_split_2 = extract_vids_from_jsonl('./highlight_train_split_2.jsonl')
    # train_vid_split_3 = extract_vids_from_jsonl('./highlight_train_split_3.jsonl')
    val_vid_list = extract_vids_from_jsonl('./highlight_val_release.jsonl')
    # test_vid_list = extract_vids_from_jsonl('./highlight_test_release.jsonl')
    
    # already_exist_val_vid_list = extract_vids_from_jsonl('./image_caption_val_8B_real.jsonl')

    # print(already_exist_val_vid_list)
    # print("len(val_vid_list) :", len(val_vid_list))
    # print("len(already_exist_val_vid_list) :", len(already_exist_val_vid_list))
    # vid_list = [vid for vid in val_vid_list if vid not in already_exist_val_vid_list]
    # print("len(vid_list) :", len(vid_list))
    save_file_name = 'val_1B_real'
    vid_list = val_vid_list
    not_exist_list = []
    cnt = 1
    #prompt = "<image>\nLook at the given image and describe the situation shown in a brief and concise sentence. Focus on the main event or activity taking place in the image without adding extra details or interpretations. Your response should only be the generated sentence, without any additional explanations or formatting."
    #prompt = '<image>\nPlease describe the image in one sentence. Focus on the main event or activity taking place in the image without adding extra details or interpretations. Your response should only be the generated sentence, without any additional explanations or formatting.'
    prompt = '<image>\nPlease describe the image in one sentence.'
    for vid_name in vid_list:
        frame_dict = {}
        vid_dict = {}
        if os.path.exists(os.path.join(frame_dir, vid_name)):
            for i in range(0, len(os.listdir(os.path.join(frame_dir, vid_name)))):
                frame_path = os.path.join(frame_dir, vid_name, vid_name + '_' + str(i) + '.jpg')
                response = make_caption(frame_path, model, tokenizer, generation_config, prompt)
                frame_dict[i] = response
            vid_dict['vid'] = vid_name
            vid_dict['captions'] = frame_dict
            
            with open(f'./image_caption_{save_file_name}.jsonl', 'a') as jsonl_file:
                json.dump(vid_dict, jsonl_file)
                jsonl_file.write('\n')
            print(cnt, vid_name)
            cnt += 1
        else:
            print(f"{vid_name} Exist? :", os.path.exists(os.path.join(frame_dir, vid_name)))
            print("Frame count :", len(os.listdir(os.path.join(frame_dir, vid_name))))
            not_exist_list.append(vid_name)
            with open(f'./not_exist_{save_file_name}.json', 'w') as json_file:
                json.dump(not_exist_list, json_file, indent=4)

def generate_summary():
    caption_file = './image_caption_train_total.jsonl'
    save_file_name = './video_summary_train_total.jsonl'
    with open(caption_file, 'r') as file:
        data = file.readlines()
    cnt = 1
    for line in data:
        l = json.loads(line)
        vid = l['vid']
        captions = l['captions']

        prompt_clustering = (
            "I have a series of 75 image frames from a video, each described by a text caption. Your task is to generate a concise video summary organized into time intervals, covering all frames from 0 to 74. The output format should ensure that the frame numbers start from 0 and end at 74, without missing any frames.\n"
            "Time Range: The range of frames for each group of similar scenes should be based on the content and should vary as appropriate (e.g., group related frames, but handle single frames that stand alone).\n"
            "Scene Summary: A brief description summarizing the main action or visual content of the frames within the time range.\n"
            "Follow these steps:\n"
            "1. **Time-based Grouping**: Group consecutive frames (by frame number) that describe similar or related scenes based on the provided captions. Make sure to include all frames from 0 to 74. If a single frame stands alone without any related frames, include it as its own time range.(e.g., 15-15).\n"
            "2. **Scene Summarization**: For each group, provide a concise, coherent summary of the scene based on the provided captions.\n"
            "3. **Output Format**: The output must be in JSON format. Each key should represent the time range (in the form 'start frame - end frame') and each value should contain a concise scene summary.\n"
            f"Here are the captions for each frame: {captions}\n"
            "Your output should be a valid JSON object, directly formatted as JSON with no additional explanation or text. Only output the JSON object, and the frame ranges should be determined based on the content.\n"
            "For example:\n"
            "{\n"
            "    '0-end_frame': 'Description of scene 1',\n"
            "    'start_frame-end_frame': 'Description of scene 2',\n"
            "    'start_frame-end_frame': 'Description of scene 3',\n"
            "    ...\n"
            "    'start_frame-74': 'Description of final scene'\n"
            "}\n"
            "Remember: Output only the JSON object. Do not include any introductory text or explanation. Every frame from 0 to 74 must be included, even if a frame stands alone."
        )

        #print(prompt)
        response = model.chat(tokenizer, None, prompt_clustering, generation_config, history=None)
        #print(response)
       
        summary_data = {
        'vid': vid,
        }
        
        try:
            # response가 JSON 형식인지 확인
            summary_data['summary'] = json.loads(response)
        except json.JSONDecodeError:
            json_start_idx = response.find("json\n")
            if json_start_idx != -1:
                response = response[json_start_idx + 6:]
            try:
                # JSON 형식이 아닐 경우 그냥 문자열로 저장
                summary_data['summary'] = json.loads(response)
            except json.JSONDecodeError:
                # JSON 형식이 아닐 경우 그냥 문자열로 저장
                summary_data['summary'] = response

        # 파일에 JSON 객체를 jsonl 형식으로 저장
        with open(save_file_name, 'a') as jsonl_file:
            json.dump(summary_data, jsonl_file)
            jsonl_file.write('\n')
        print(cnt, vid)
        cnt +=1
if __name__ == "__main__":
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    path = "OpenGVLab/InternVL2-1B"
    
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    # 함수 호출
    generate_captions()
    #generate_summary()