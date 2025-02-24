import os
import yt_dlp

# Définition des requêtes de recherche avec "no commentary"
game_queries = {
    'starcraft': 'starcraft gameplay no commentary',
    'lol': 'league of legends gameplay no commentary',
    'cod': 'call of duty gameplay no commentary'
}

output_folder = "videos"

# Fonction pour créer le dossier de sortie du jeu
def ensure_folder(game):
    game_folder = os.path.join(output_folder, game)
    os.makedirs(game_folder, exist_ok=True)
    return game_folder

def get_video_urls(search_query, min_videos=15, max_duration=1800, min_duration=300):
    """ Récupère au moins `min_videos` vidéos respectant les critères """
    video_urls = []
    max_results = min_videos * 2
    
    while len(video_urls) < min_videos:
        search_url = f"ytsearch{max_results}:{search_query}"
        ydl_opts = {
            'quiet': True,
            'default_search': 'ytsearch',
            'extract_flat': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
        
        if not info or 'entries' not in info:
            print(f"Aucune vidéo trouvée pour {search_query}.")
            return []

        for entry in info.get('entries', []):
            if entry and 'url' in entry:
                duration = entry.get('duration', 0)
                
                if min_duration <= duration <= max_duration:
                    video_urls.append(entry['url'])

            if len(video_urls) >= min_videos:
                break

        max_results *= 2  # Augmente le nombre de résultats si on n'a pas assez de vidéos

    return video_urls[:min_videos]  # Garde exactement `min_videos`

def download_videos(video_urls, game):
    """ Télécharge les vidéos sélectionnées dans le bon dossier """
    game_folder = ensure_folder(game)
    ydl_opts = {
        'format': 'bv*[height<=480][ext=mp4]',
        'outtmpl': f'{game_folder}/{game}_%(id)s.%(ext)s',
        'noplaylist': True,
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(video_urls)

# Télécharger les vidéos filtrées
for game, query in game_queries.items():
    print(f"Recherche d'au moins 15 vidéos pour {game}...")
    video_urls = get_video_urls(query, min_videos=15)
    
    if video_urls:
        print(f"Téléchargement de {len(video_urls)} vidéos pour {game}...")
        download_videos(video_urls, game)
    else:
        print(f"Pas assez de vidéos trouvées pour {game}.")
