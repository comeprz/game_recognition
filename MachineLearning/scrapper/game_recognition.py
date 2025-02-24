import cv2
import random
import os
import glob

video_folder = "videos"
output_folder = "screenshots"
num_screenshots = 1000

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lister tous les jeux (chaque dossier dans 'video_folder' est un jeu)
games = [d for d in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, d))]

for game in games:
    game_video_paths = glob.glob(os.path.join(video_folder, game, "*.mp4"))  # Toutes les vidéos du jeu
    if not game_video_paths:
        print(f"Aucune vidéo trouvée pour {game}, passage au suivant...")
        continue

    game_folder = os.path.join(output_folder, game)
    os.makedirs(game_folder, exist_ok=True)

    # Répartir les screenshots sur toutes les vidéos du jeu
    total_videos = len(game_video_paths)
    screenshots_per_video = num_screenshots // total_videos  # Screenshots à prendre par vidéo

    for video_path in game_video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur : Impossible d'ouvrir la vidéo {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Traitement de {video_path} ({total_frames} frames)...")

        for i in range(screenshots_per_video):
            frame_id = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (227, 128))
                cv2.imwrite(f"{game_folder}/frame_{i+1}_{os.path.basename(video_path)}.jpg", frame)

        cap.release()

print("Extraction terminée.")
