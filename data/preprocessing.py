import torch
import os
import numpy as np
import ffmpeg
import math
import glob
import json
import cv2

class VideoLoader:
    """Pytorch video loader optimized for frame extraction."""
    
    def __init__(self, framerate=1/2, size=224, centercrop=True):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def _get_video_info(self, video_path):
        """Get basic video information."""
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(self.convert_to_float(video_stream['avg_frame_rate']))
        frames_length = int(video_stream.get('nb_frames', -1))
        duration = float(video_stream.get('duration', -1))
        return {"duration": duration, "frames_length": frames_length, "fps": fps, "height": height, "width": width}

    def convert_to_float(self, frac_str):
        """Convert fraction or string to float."""
        try:
            return float(frac_str)
        except ValueError:
            num, denom = frac_str.split('/')
            return float(num) / float(denom)

    def _get_output_dim(self, h, w):
        """Get output dimensions based on centercrop and size."""
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    # deprecated
    def read_video_from_file_and_save(self, video_path, output_dir, image_name):
        """Extract and save frames from the video."""
        info = self._get_video_info(video_path)
        height, width = self._get_output_dim(info["height"], info["width"])
        fps = self.framerate
        duration = info["duration"]

        if duration < 0:
            print(f"Invalid video duration for {video_path}")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Calculate the number of frames to extract
        total_frames = int(duration * fps)

        # Iterate and save each frame
        for i in range(total_frames):
            out_file = os.path.join(output_dir, f'{image_name}_{i}.jpg')

            # Seek to the correct time and extract the frame
            (
                ffmpeg
                .input(video_path, ss=i / fps)
                .filter('scale', width, height)
                .output(out_file, vframes=1, format='image2', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )

        print(f"Frames saved in {output_dir}")
   

    def read_video_and_save_frames(self, video_path, output_dir, image_name):
        try:
            info = self._get_video_info(video_path)
            h, w = info["height"], info["width"]
        except Exception:
            print('ffprobe failed at: {}'.format(video_path))
            return {'video': None, 'input': video_path, 'info': {}}
        
        height, width = self._get_output_dim(h, w)
        try:
            duration = info["duration"]
            fps = self.framerate
            if duration > 0 and duration < 1 / fps + 0.1:
                fps = 2 / max(int(duration), 1)
                print(duration, fps)
        except Exception:
            fps = self.framerate
        
        # ffmpeg 명령어를 통해 비디오 프레임 읽기
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)
            .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size

        # 프레임 데이터를 numpy 배열로 변환
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        
        # 프레임별로 이미지 파일로 저장
        frame_paths = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, frame in enumerate(video):
            frame_path = os.path.join(output_dir, f'{image_name}_{i}.jpg')
            cv2.imwrite(frame_path, frame[:, :, ::-1])  # RGB에서 BGR로 변환하여 저장
            frame_paths.append(frame_path)

        return {'frames': frame_paths, 'input': video_path, 'info': info}
    def save_frames_from_output(self, output_data, output_dir, vid_name):
        # 출력된 raw 비디오 데이터를 numpy 배열로 변환
        # 비디오 로드 시 설정된 높이와 너비를 사용해야 합니다.
        height, width = self.size, self.size
        video = np.frombuffer(output_data, np.uint8).reshape([-1, height, width, 3])

        # 출력 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 프레임을 개별 이미지 파일로 저장
        frame_paths = []
        for i, frame in enumerate(video):
            frame_path = os.path.join(output_dir, f'{vid_name}_{i}.jpg')
            cv2.imwrite(frame_path, frame[:, :, ::-1])  # RGB -> BGR 변환
            frame_paths.append(frame_path)
        print(f"Frames saved in {output_dir}")
        return frame_paths
def get_mp4_filenames_without_extension(directory):
    filenames = []
    
    # 지정된 디렉토리에서 파일 리스트 가져오기
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            # 확장자를 제거한 파일 이름 저장
            filenames.append(os.path.splitext(file)[0])
    
    return filenames

def get_mp4_file_paths(directory):
    # 경로 내 모든 mp4 파일 찾기 (하위 폴더 포함)
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'), recursive=True)
    return mp4_files

def extract_vids_from_jsonl(file_path):
    vid_list = []
    
    # 파일을 열고 각 줄을 처리
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)  # json 형식으로 변환
            vid_list.append(data['vid'])  # vid 값 추출 후 리스트에 추가
    vid_list = list(set(vid_list))
    return vid_list

import os
import concurrent.futures
import ffmpeg


def process_video(video_loader, vid_path, output_dir, gpu_id, vid_name):
    # GPU ID를 사용해 ffmpeg 명령어에 hwaccel 옵션 추가
    cmd = (
        ffmpeg
        .input(vid_path)
        .filter('fps', fps=video_loader.framerate)
        .filter('scale', video_loader.size, video_loader.size)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .global_args('-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id))  # CUDA 하드웨어 가속 사용 및 GPU ID 설정
    )

    out, _ = cmd.run(capture_stdout=True, quiet=True)

    # 프레임 데이터를 저장
    video_loader.save_frames_from_output(out, output_dir, vid_name)

if __name__ == "__main__":
    video_loader = VideoLoader(framerate=1/2, size=224, centercrop=True)
    directory = '/data/Shared_Data/qvhighlights/origin_videos'
    vid_list = extract_vids_from_jsonl('./highlight_all.jsonl')

    # GPU ID 순환을 위한 설정
    num_gpus = 4

    # 멀티프로세싱 설정
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(len(vid_list)):
            vid_name = vid_list[i]
            vid_path = os.path.join(directory, vid_name + '.mp4')
            output_dir = os.path.join('/data/Shared_Data/qvhighlights/video_frames_real/', vid_name)

            # GPU ID를 순환하여 할당
            gpu_id = i % num_gpus

            # 비디오 처리 작업을 비동기적으로 실행
            futures.append(executor.submit(process_video, video_loader, vid_path, output_dir, gpu_id, vid_name))

        # 모든 작업 완료 대기
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f'Error processing video: {e}')