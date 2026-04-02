"""
Gradio Web Interface for the Trend-Aware Music Discovery Environment.

Provides an interactive demo where users can:
  1. Select a task difficulty and reset the environment
  2. Browse available songs and manually recommend one
  3. Watch the simulated user react in real-time
  4. Run the baseline agent automatically
  5. See the final grade at the end of the episode
"""

import json
import random
import gradio as gr

try:
    from server.music_discovery_env_environment import (
        MusicDiscoveryEnvironment, grade, baseline_agent, REAL_SONGS_DB
    )
    from models import MusicDiscoveryAction
except ImportError:
    from .server.music_discovery_env_environment import (
        MusicDiscoveryEnvironment, grade, baseline_agent, REAL_SONGS_DB
    )
    from .models import MusicDiscoveryAction


# ── Emoji helpers ─────────────────────────────────────────────────────────
REACTION_EMOJI = {
    "shared":           "🔥 Shared",
    "saved":            "💾 Saved",
    "added_to_playlist":"🎶 Added to Playlist",
    "played_once":      "▶️  Played Once",
    "skipped":          "⏭️  Skipped",
    "ignored":          "🚫 Ignored (repeat)",
}

VIBE_EMOJI = {
    "hyped": "⚡", "relaxed": "🌊", "party": "🎉", "focus": "🎯", "sad": "😢"
}

MEDIA_EMOJI = {
    "anime": "🎌", "game": "🎮", "movie": "🎬", "tv_show": "📺", "viral_video": "📱"
}


# ── Global session state ──────────────────────────────────────────────────
_env = MusicDiscoveryEnvironment()
_obs = None
_trajectory = []
_done = False


def _format_song_table(songs: list, recommended: list) -> str:
    """Build a markdown table of available songs."""
    rows = ["| # | ID | Title | Artist | Source | Genre | Vibe | Velocity | Age |",
            "|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(songs, 1):
        used = "~~" if s["id"] in recommended else ""
        vibe_icon = VIBE_EMOJI.get(s.get("vibe", ""), "")
        media_icon = MEDIA_EMOJI.get(s.get("media_type", ""), "")
        rows.append(
            f"| {i} | {used}{s['id']}{used} | {s['title']} | {s['artist']} "
            f"| {media_icon} {s['source_media']} | {s['genre']} "
            f"| {vibe_icon} {s.get('vibe','')} | {s['trend_velocity']} "
            f"| {s['trend_age_days']}d |"
        )
    return "\n".join(rows)


def _format_user_card(obs_dict: dict) -> str:
    """Render the user profile as a markdown card."""
    user = obs_dict.get("user", {})
    tp   = user.get("taste_profile", {})
    genres = ", ".join(tp.get("genres", [])) or "❓ Unknown"
    media  = ", ".join(tp.get("media_interests", [])) or "❓ Unknown"
    openness = user.get("discovery_openness", 0.5)
    last_3   = obs_dict.get("last_3_reactions", [])
    reactions_str = ", ".join(REACTION_EMOJI.get(r, r) for r in last_3) if last_3 else "None yet"

    bar = "█" * int(openness * 10) + "░" * (10 - int(openness * 10))
    return (
        f"### 👤 User Profile\n"
        f"- **Genres:** {genres}\n"
        f"- **Media interests:** {media}\n"
        f"- **Discovery openness:** `[{bar}]` {openness:.1f}\n"
        f"- **Recent reactions:** {reactions_str}\n"
        f"- **Step:** {obs_dict.get('step_count', 0)} / 10\n"
    )


def _format_trajectory(traj: list) -> str:
    """Render the trajectory log as a feed."""
    if not traj:
        return "*No history yet. Start recommending!*"
    
    html = '<div style="display: flex; flex-direction: column; gap: 8px;">'
    for t in reversed(traj):
        emoji = REACTION_EMOJI.get(t.get("reaction", ""), t.get("reaction", ""))
        reward = t.get("reward", 0)
        color = "#1DB954" if reward > 0.5 else ("#ffcc00" if reward > 0 else "#ff4444")
        html += f'''
        <div style="background: #282828; padding: 12px; border-radius: 8px; border-left: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: white;">Step {t.get("step","")} • {t.get("song_id","")}</span>
                <span style="color: {color}; font-weight: bold;">{reward:+.2f}</span>
            </div>
            <div style="color: #b3b3b3; font-size: 0.9em; margin-top: 4px;">{emoji} • {t.get("trend_age_days","")}d trend</div>
        </div>
        '''
    html += '</div>'
    return html


# ── Gradio callbacks ─────────────────────────────────────────────────────

def reset_env(task: str):
    global _env, _obs, _trajectory, _done
    _env = MusicDiscoveryEnvironment()
    _obs = _env.reset(task=task)
    _trajectory = []
    _done = False
    obs_dict = _obs.model_dump()

    song_ids = [s["id"] for s in obs_dict.get("trending_songs", [])]
    user_card = _format_user_card(obs_dict)
    song_table = _format_song_table(obs_dict.get("trending_songs", []), [])
    traj_log = _format_trajectory([])
    status = f"✅ Environment reset — **{task.upper()}** task loaded. Pick a song!"

    return (
        user_card,
        song_table,
        traj_log,
        status,
        gr.Dropdown(choices=song_ids, value=song_ids[0] if song_ids else None),
        "",  # grade
    )


def step_env(song_id: str):
    global _obs, _trajectory, _done
    if _done:
        return (
            _format_user_card(_obs.model_dump()),
            _format_song_table(_obs.model_dump().get("trending_songs", []),
                               _obs.model_dump().get("recommended_history", [])),
            _format_trajectory(_trajectory),
            "⚠️ Episode is done! Reset to start a new one.",
            gr.Dropdown(),
            f"**Final Grade: {grade(_trajectory):.2f}**",
        )
    if not song_id:
        return (gr.Dropdown(),) * 5 + ("Select a song first!",)

    _obs = _env.step(MusicDiscoveryAction(song_id=song_id))
    obs_dict = _obs.model_dump()
    _done = _obs.done

    if obs_dict.get("session_engagement"):
        _trajectory.append(obs_dict["session_engagement"][-1])

    last_info = _trajectory[-1] if _trajectory else {}
    reaction = REACTION_EMOJI.get(last_info.get("reaction", ""), last_info.get("reaction", ""))
    reward = last_info.get("reward", 0)

    status = f"Step {obs_dict['step_count']}: **{song_id}** → {reaction} (reward: {reward:+.2f})"
    if _done:
        final = grade(_trajectory)
        status += f"\n\n🏁 **Episode complete! Final Grade: {final:.2f}**"
        grade_str = f"## 🏆 Final Grade: {final:.2f}"
    else:
        grade_str = ""

    avail = [s["id"] for s in obs_dict.get("trending_songs", [])
             if s["id"] not in obs_dict.get("recommended_history", [])]

    return (
        _format_user_card(obs_dict),
        _format_song_table(obs_dict.get("trending_songs", []),
                           obs_dict.get("recommended_history", [])),
        _format_trajectory(_trajectory),
        status,
        gr.Dropdown(choices=avail, value=avail[0] if avail else None),
        grade_str,
    )


def run_baseline_demo(task: str):
    """Run the full baseline agent and return the completed trajectory."""
    global _env, _obs, _trajectory, _done
    random.seed(42)
    _env = MusicDiscoveryEnvironment()
    _obs = _env.reset(task=task)
    _trajectory = []
    _done = False
    obs_dict = _obs.model_dump()

    while not _done:
        action = baseline_agent(obs_dict)
        _obs = _env.step(MusicDiscoveryAction(song_id=action["song_id"]))
        obs_dict = _obs.model_dump()
        _done = _obs.done
        if obs_dict.get("session_engagement"):
            _trajectory.append(obs_dict["session_engagement"][-1])

    final = grade(_trajectory)
    status = f"🤖 Baseline agent completed **{task.upper()}** task — Grade: **{final:.2f}**"

    avail = [s["id"] for s in obs_dict.get("trending_songs", [])
             if s["id"] not in obs_dict.get("recommended_history", [])]

    return (
        _format_user_card(obs_dict),
        _format_song_table(obs_dict.get("trending_songs", []),
                           obs_dict.get("recommended_history", [])),
        _format_trajectory(_trajectory),
        status,
        gr.Dropdown(choices=avail, value=avail[0] if avail else None),
        f"## 🤖 Baseline Grade: {final:.2f}",
    )


# ── Build the Gradio UI ──────────────────────────────────────────────────

def create_gradio_app():
    """Build the Spotify-style Gradio interface."""
    with gr.Blocks(
        title="🎧 Spotify Discovery RL",
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
        css="""
        .gradio-container { background-color: #121212 !important; color: white !important; }
        .sidebar { background-color: #000000; padding: 20px; border-radius: 12px; border: 1px solid #282828; }
        .main-content { background-color: #121212; padding: 10px; }
        .card { background-color: #181818; padding: 15px; border-radius: 12px; border: 1px solid #282828; margin-bottom: 15px; }
        .spotify-btn { background-color: #1DB954 !important; color: black !important; font-weight: bold !important; border-radius: 50px !important; }
        .spotify-btn:hover { background-color: #1ed760 !important; }
        .secondary-btn { background-color: transparent !important; color: white !important; border: 1px solid #b3b3b3 !important; border-radius: 50px !important; }
        table { width: 100%; border-collapse: collapse; background: #181818; border-radius: 8px; overflow: hidden; }
        th { background: #282828; color: #b3b3b3; text-align: left; padding: 12px; font-size: 0.8em; text-transform: uppercase; }
        td { border-top: 1px solid #282828; padding: 12px; color: white; font-size: 0.9em; }
        tr:hover td { background: #282828; }
        h1, h2, h3 { color: white !important; }
        .muted { color: #b3b3b3 !important; font-size: 0.9em; }
        """
    ) as demo:
        with gr.Row():
            # --- Sidebar ---
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.HTML('<div style="font-size: 24px; font-weight: bold; margin-bottom: 20px;">🎧 Discovery</div>')
                
                with gr.Group():
                    task_dd = gr.Dropdown(
                        choices=["easy", "medium", "hard"],
                        value="easy", label="Difficulty", container=False
                    )
                    reset_btn = gr.Button("🔄 Reset Session", variant="primary", elem_classes="spotify-btn")
                    baseline_btn = gr.Button("🤖 Run Autopilot", variant="secondary", elem_classes="secondary-btn")
                
                gr.HTML('<div style="margin-top: 30px;"></div>')
                user_md = gr.Markdown("### 👤 Listener\n*Reset to load profile.*", elem_classes="card")
                grade_md = gr.HTML("", elem_classes="card")

            # --- Main Content ---
            with gr.Column(scale=3, elem_classes="main-content"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('<h1 style="margin:0;">Discover Weekly</h1><p class="muted">Your personalized RL recommendation benchmark.</p>')
                    with gr.Column(scale=1):
                        status_md = gr.Markdown("*Ready to start.*", elem_classes="card")

                with gr.Row():
                    # --- Selection Column ---
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="card"):
                            gr.HTML('<div style="font-weight: bold; margin-bottom: 10px;">Recommended for you</div>')
                            song_dd = gr.Dropdown(
                                choices=[], label="", show_label=False, container=False, interactive=True
                            )
                            step_btn = gr.Button("🎶 Recommend Now", variant="primary", elem_classes="spotify-btn")

                    # --- History Column ---
                    with gr.Column(scale=1):
                        gr.HTML('<div style="font-weight: bold; margin-bottom: 10px;">Recent Activity</div>')
                        traj_md = gr.HTML("*No history yet.*", elem_classes="card")

                gr.HTML('<div style="margin-top: 20px;"></div>')
                gr.HTML('<div style="font-weight: bold; margin-bottom: 10px;">Available in Catalogue</div>')
                songs_md = gr.Markdown("*Load session to view songs.*")

        outputs = [user_md, songs_md, traj_md, status_md, song_dd, grade_md]

        reset_btn.click(
            fn=reset_env, inputs=[task_dd], outputs=outputs
        )
        step_btn.click(
            fn=step_env, inputs=[song_dd], outputs=outputs
        )
        baseline_btn.click(
            fn=run_baseline_demo, inputs=[task_dd], outputs=outputs
        )

    return demo


# When imported, expose the app for mounting
gradio_app = create_gradio_app()
