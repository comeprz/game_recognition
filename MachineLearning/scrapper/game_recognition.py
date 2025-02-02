import cv2
import random
import os

# Vidéos
video_files = {
    'fortnite': 'videos/fortnite1.mp4',  
    'lol': 'videos/lol1.mp4',    
    'cod': 'videos/cod1.mp4'
}

output_folder = 'screenshots'

# Nombre de screens
num_screenshots = 100

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parcourir chaque vidéo
for game, video_path in video_files.items():
    game_folder = os.path.join(output_folder, game)
    if not os.path.exists(game_folder):
        os.makedirs(game_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Traitement de la vidéo {game} ({total_frames} frames)...")

    for i in range(num_screenshots):
        frame_id = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'{game_folder}/frame_{i+1}.jpg', frame)
            print(f'Screenshot {i+1} pour {game} sauvegardé.')

    cap.release()

print("Extraction terminée")