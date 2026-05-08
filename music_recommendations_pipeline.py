"""
Weekly Music Recommendation Pipeline - Pure KFP SDK v2 Implementation

This pipeline:
- Loads user listening history and song catalog from GitHub
- Trains a collaborative filtering recommendation model (LightFM)
- Generates personalized playlists for users
- Evaluates playlist quality

Pipeline DAG:
load_data → build_matrix → train_model → ┬ detect_drift
                                         └ generate_playlists → evaluate
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Metrics


PACKAGES = ['pandas==2.0.3', 'numpy==1.24.3', 'scipy==1.11.1', 
            'scikit-learn==1.3.0', 'lightfm==1.17']
BASE_IMAGE = 'python:3.11'


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def load_data(
    listening_data_out: Output[Dataset],
    songs_out: Output[Dataset],
    metrics: Output[Metrics]
):
    """Load listening events and song catalog, calculate implicit ratings."""
    import pandas as pd
    
    def calculate_implicit_rating(play_count, completion_rate, skip_count, save_count):
        base_score = play_count * 0.5
        completion_bonus = completion_rate * 3
        skip_penalty = skip_count * -1.5
        save_bonus = save_count * 2
        rating = base_score + completion_bonus + skip_penalty + save_bonus
        return max(0, min(10, rating))
    
    DATA_URL = "https://raw.githubusercontent.com/ada333/Kale-music-recommendation-use-case/main/data"
    listening_data = pd.read_csv(f"{DATA_URL}/listening_events.csv")
    songs = pd.read_csv(f"{DATA_URL}/songs.csv")
    
    listening_data['rating'] = listening_data.apply(
        lambda row: calculate_implicit_rating(
            row['play_count'], 
            row['completion_rate'], 
            row['skip_count'], 
            row['save_count']
        ), 
        axis=1
    )
    
    n_users = listening_data['user_id'].nunique()
    n_songs = listening_data['song_id'].nunique()
    
    listening_data.to_parquet(listening_data_out.path)
    songs.to_parquet(songs_out.path)
    
    metrics.log_metric("n-users", n_users)
    metrics.log_metric("n-songs-listened", n_songs)
    print(f"Loaded {n_users} users, {n_songs} songs")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def build_matrix(
    listening_data_in: Input[Dataset],
    songs_in: Input[Dataset],
    interaction_matrix_out: Output[Dataset],
    user_ids_out: Output[Dataset],
    song_to_idx_out: Output[Dataset],
    listening_data_out: Output[Dataset],
    songs_out: Output[Dataset]
):
    """Build user-song interaction matrix for collaborative filtering."""
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix, save_npz
    import json
    
    listening_data = pd.read_parquet(listening_data_in.path)
    songs = pd.read_parquet(songs_in.path)
    
    user_ids = listening_data['user_id'].unique()
    song_ids = songs['song_id'].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    song_to_idx = {song_id: idx for idx, song_id in enumerate(song_ids)}
    
    rows, cols, data = [], [], []
    for _, row in listening_data.iterrows():
        rows.append(user_to_idx[row['user_id']])
        cols.append(song_to_idx[row['song_id']])
        data.append(row['rating'])
    
    interaction_matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(len(user_ids), len(song_ids))
    )
    
    save_npz(interaction_matrix_out.path, interaction_matrix)
    np.save(user_ids_out.path, user_ids)
    with open(song_to_idx_out.path, 'w') as f:
        json.dump({str(k): v for k, v in song_to_idx.items()}, f)
    
    listening_data.to_parquet(listening_data_out.path)
    songs.to_parquet(songs_out.path)
    
    print(f"Built {interaction_matrix.shape[0]}x{interaction_matrix.shape[1]} interaction matrix")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def train_model(
    interaction_matrix_in: Input[Dataset],
    song_to_idx_in: Input[Dataset],
    user_embeddings_out: Output[Dataset],
    song_embeddings_out: Output[Dataset],
    idx_to_song_out: Output[Dataset],
    metrics: Output[Metrics]
):
    """Train LightFM collaborative filtering model."""
    import numpy as np
    from scipy.sparse import load_npz
    from lightfm import LightFM
    from lightfm.evaluation import precision_at_k
    import json
    
    interaction_matrix = load_npz(interaction_matrix_in.path)
    
    with open(song_to_idx_in.path, 'r') as f:
        song_to_idx = {int(k): v for k, v in json.load(f).items()}
    
    model = LightFM(no_components=32, loss='warp', random_state=42)
    for _ in range(10):
        model.fit_partial(interaction_matrix, epochs=1)
    
    user_embeddings = model.user_embeddings
    song_embeddings = model.item_embeddings
    idx_to_song = {idx: song_id for song_id, idx in song_to_idx.items()}
    
    final_precision = float(precision_at_k(model, interaction_matrix, k=10).mean())
    
    np.save(user_embeddings_out.path, user_embeddings)
    np.save(song_embeddings_out.path, song_embeddings)
    with open(idx_to_song_out.path, 'w') as f:
        json.dump({str(k): v for k, v in idx_to_song.items()}, f)
    
    metrics.log_metric("final-precision", final_precision)
    print(f"Model trained, precision@10: {final_precision:.3f}")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def detect_drift(
    listening_data_in: Input[Dataset],
    songs_in: Input[Dataset],
    metrics: Output[Metrics]
):
    """Check for genre distribution drift compared to historical baseline."""
    import pandas as pd
    from datetime import datetime
    
    listening_data = pd.read_parquet(listening_data_in.path)
    songs = pd.read_parquet(songs_in.path)
    
    current_genre_dist = listening_data.merge(songs, on='song_id')['genre'].value_counts(normalize=True)
    
    historical_baseline = {
        'Rock': 0.22,
        'Electronic': 0.20,
        'Rap': 0.18,
        'Czech songs': 0.12
    }
    
    max_shift = 0
    for genre, historical in historical_baseline.items():
        current = current_genre_dist.get(genre, 0)
        shift = abs(current - historical) * 100
        max_shift = max(max_shift, shift)
    
    drift_status = 'NORMAL'
    if max_shift > 15:
        drift_status = 'MAJOR_DRIFT'
    elif max_shift > 8:
        drift_status = 'MODERATE_DRIFT'
    
    metrics.log_metric("max-genre-shift", max_shift)
    print(f"Maximum genre shift: {max_shift:.1f}%, status: {drift_status}")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def generate_playlists(
    user_embeddings_in: Input[Dataset],
    song_embeddings_in: Input[Dataset],
    idx_to_song_in: Input[Dataset],
    user_ids_in: Input[Dataset],
    listening_data_in: Input[Dataset],
    songs_in: Input[Dataset],
    playlists_out: Output[Dataset]
):
    """Generate personalized playlists using learned embeddings."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import json
    
    user_embeddings = np.load(user_embeddings_in.path)
    song_embeddings = np.load(song_embeddings_in.path)
    user_ids = np.load(user_ids_in.path, allow_pickle=True)
    listening_data = pd.read_parquet(listening_data_in.path)
    songs = pd.read_parquet(songs_in.path)
    
    with open(idx_to_song_in.path, 'r') as f:
        idx_to_song = {int(k): v for k, v in json.load(f).items()}
    
    NUM_DEMO_USERS = 10
    PLAYLIST_SIZE = 30
    NUM_CANDIDATES = 100
    all_playlists = {}
    
    for user_idx, user_id in enumerate(user_ids[:NUM_DEMO_USERS]):
        user_emb = user_embeddings[user_idx].reshape(1, -1)
        similarities = cosine_similarity(user_emb, song_embeddings)[0]
        
        user_history = listening_data[listening_data['user_id'] == user_id]['song_id'].values
        user_listened = songs[songs['song_id'].isin(user_history)]
        language_counts = user_listened['language'].value_counts()
        non_instrumental = language_counts[language_counts.index != 'Instrumental']
        user_language = non_instrumental.index[0] if len(non_instrumental) > 0 else 'English'
        
        candidates = []
        for song_idx, similarity in enumerate(similarities):
            song_id = idx_to_song[song_idx]
            if song_id not in user_history:
                song_info = songs[songs['song_id'] == song_id].iloc[0]
                candidates.append({
                    'song_id': int(song_id),
                    'title': song_info['title'],
                    'artist': song_info['artist'],
                    'genre': song_info['genre'],
                    'language': song_info['language'],
                    'similarity': float(similarity)
                })
        
        candidates_df = pd.DataFrame(candidates).nlargest(NUM_CANDIDATES, 'similarity')
        
        filtered = candidates_df[
            (candidates_df['language'] == user_language) |
            (candidates_df['language'] == 'Instrumental')
        ]
        if len(filtered) < PLAYLIST_SIZE:
            filtered = candidates_df
        
        final_playlist = filtered.head(PLAYLIST_SIZE)
        
        all_playlists[str(user_id)] = {
            'songs': final_playlist.to_dict('records'),
            'diversity_score': int(final_playlist['genre'].nunique()),
            'avg_similarity': float(final_playlist['similarity'].mean()),
            'user_language': user_language
        }
    
    with open(playlists_out.path, 'w') as f:
        json.dump(all_playlists, f)
    
    print(f"Generated {len(all_playlists)} playlists with {PLAYLIST_SIZE} songs each")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def evaluate(
    playlists_in: Input[Dataset],
    metrics: Output[Metrics]
):
    """
    Evaluate playlist quality.
    Metrics: genre diversity, recommendation similarity, filter bubble rate.
    """
    import pandas as pd
    import numpy as np
    import json
    
    with open(playlists_in.path, 'r') as f:
        all_playlists = json.load(f)
    
    diversity_scores = [p['diversity_score'] for p in all_playlists.values()]
    similarity_scores = [p['avg_similarity'] for p in all_playlists.values()]
    
    low_diversity_count = 0
    for playlist in all_playlists.values():
        songs_df = pd.DataFrame(playlist['songs'])
        top_genre_pct = songs_df['genre'].value_counts().iloc[0] / len(songs_df)
        if top_genre_pct > 0.7:
            low_diversity_count += 1
    
    filter_bubble_rate = low_diversity_count / len(all_playlists)
    avg_diversity = float(np.mean(diversity_scores))
    avg_similarity = float(np.mean(similarity_scores))
    
    all_checks_passed = (
        avg_diversity >= 5.0 and
        filter_bubble_rate < 0.15 and
        avg_similarity >= 0.75
    )
    
    metrics.log_metric("avg-diversity", avg_diversity)
    metrics.log_metric("avg-similarity", avg_similarity)
    metrics.log_metric("filter-bubble-rate", filter_bubble_rate)
    
    print(f"Quality check: {'PASSED' if all_checks_passed else 'FAILED'}")
    print(f"avg-diversity: {avg_diversity}")
    print(f"avg-similarity: {avg_similarity}")
    print(f"filter-bubble-rate: {filter_bubble_rate}")


@dsl.pipeline(
    name='weekly-music-recommendations',
    description='Collaborative filtering pipeline for personalized music playlists'
)
def music_recommendation_pipeline():
    """
    Pipeline DAG:
    load_data → build_matrix → train_model → ┬ detect_drift
                                             └ generate_playlists → evaluate
    """
    load_data_task = load_data()
    
    build_matrix_task = build_matrix(
        listening_data_in=load_data_task.outputs['listening_data_out'],
        songs_in=load_data_task.outputs['songs_out']
    )
    
    train_model_task = train_model(
        interaction_matrix_in=build_matrix_task.outputs['interaction_matrix_out'],
        song_to_idx_in=build_matrix_task.outputs['song_to_idx_out']
    )
    
    # detect_drift runs in parallel with generate_playlists
    detect_drift_task = detect_drift(
        listening_data_in=build_matrix_task.outputs['listening_data_out'],
        songs_in=build_matrix_task.outputs['songs_out']
    )
    
    generate_playlists_task = generate_playlists(
        user_embeddings_in=train_model_task.outputs['user_embeddings_out'],
        song_embeddings_in=train_model_task.outputs['song_embeddings_out'],
        idx_to_song_in=train_model_task.outputs['idx_to_song_out'],
        user_ids_in=build_matrix_task.outputs['user_ids_out'],
        listening_data_in=build_matrix_task.outputs['listening_data_out'],
        songs_in=build_matrix_task.outputs['songs_out']
    )
    
    evaluate_task = evaluate(
        playlists_in=generate_playlists_task.outputs['playlists_out']
    )


if __name__ == "__main__":
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=music_recommendation_pipeline,
        package_path='music_recommendations_pipeline.yaml'
    )
    print("Pipeline compiled to music_recommendations_pipeline.yaml")
