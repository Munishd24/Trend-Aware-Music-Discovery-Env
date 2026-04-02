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

    global_mood = obs_dict.get("global_mood_trend", "Unknown")
    mood_emoji = VIBE_EMOJI.get(global_mood, "🌐")

    return (
        f"<div style='display: flex; flex-direction: column; gap: 8px;'>"
        f"  <div style='background: linear-gradient(135deg, rgba(29,185,84,0.15) 0%, rgba(0,0,0,0.5) 100%); padding: 12px; border-radius: 12px; border: 1px solid rgba(29,185,84,0.3); text-align: center; margin-bottom: 10px;'>"
        f"    <div style='font-size: 0.8em; color: #1ed760; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;'>🌍 Global Viral Mood</div>"
        f"    <div style='font-size: 1.5em; font-weight: 800; color: #fff; text-shadow: 0 0 10px rgba(29,185,84,0.6);'>{mood_emoji} {global_mood.capitalize()}</div>"
        f"  </div>"
        f"  <h3 style='margin: 0 0 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;'>👤 User Profile</h3>"
        f"  <div style='display: flex; gap: 8px;'><span style='color: #9ca3af; min-width: 80px;'>Genres</span> <span style='font-weight: 600; color: #fff;'>{genres}</span></div>"
        f"  <div style='display: flex; gap: 8px;'><span style='color: #9ca3af; min-width: 80px;'>Media</span> <span style='font-weight: 600; color: #fff;'>{media}</span></div>"
        f"  <div style='display: flex; gap: 8px; align-items: center;'><span style='color: #9ca3af; min-width: 80px;'>Openness</span> <div style='height: 6px; width: 100px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden;'><div style='height: 100%; width: {int(openness*100)}%; background: #1ed760; box-shadow: 0 0 8px #1ed760;'></div></div><span style='font-size: 0.85em; margin-left: 6px; color: #1ed760; font-weight: bold;'>{openness:.1f}</span></div>"
        f"  <div style='display: flex; gap: 8px; margin-top: 5px;'><span style='color: #9ca3af; min-width: 80px;'>Reactions</span> <span style='font-size: 0.9em; background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 6px; color: #ccc;'>{reactions_str}</span></div>"
        f"  <div style='margin-top: 15px; text-align: center; color: #fff; background: rgba(29,185,84,0.1); padding: 8px; border-radius: 12px; border: 1px solid rgba(29,185,84,0.3); font-weight: 600;'>"
        f"    Step {obs_dict.get('step_count', 0)} / 10"
        f"  </div>"
        f"</div>"
    )


def _format_trajectory(traj: list) -> str:
    """Render the trajectory log as a feed."""
    if not traj:
        return "*No history yet. Start recommending!*"
    
    html = '<div style="display: flex; flex-direction: column; gap: 12px;">'
    for t in reversed(traj):
        emoji = REACTION_EMOJI.get(t.get("reaction", ""), t.get("reaction", ""))
        reward = t.get("reward", 0)
        color = "#1ed760" if reward > 0.5 else ("#facc15" if reward > 0 else "#ef4444")
        bg_rgb = "29, 185, 84" if reward > 0.5 else ("250, 204, 21" if reward > 0 else "239, 68, 68")
        html += f'''
        <div style="background: rgba({bg_rgb}, 0.05); padding: 16px; border-radius: 12px; border: 1px solid rgba({bg_rgb}, 0.2); border-left: 4px solid {color}; transition: transform 0.2s, background 0.2s;" onmouseover="this.style.transform='translateX(4px)'; this.style.background='rgba({bg_rgb}, 0.08)';" onmouseout="this.style.transform='translateX(0)'; this.style.background='rgba({bg_rgb}, 0.05)';">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #fff; font-weight: 600; font-size: 1.05em; display: flex; align-items: center; gap: 8px;">
                    <span style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 20px; font-size: 0.8em; color: #ccc;">{t.get("step","")}</span> 
                    {t.get("song_id","")}
                </span>
                <span style="color: {color}; font-weight: 800; text-shadow: 0 0 10px rgba({bg_rgb}, 0.4);">{reward:+.2f}</span>
            </div>
            <div style="color: #a0a0a0; font-size: 0.9em; margin-top: 8px; display: flex; gap: 10px;">
                <span style="background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px;">{emoji}</span>
                <span style="background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px;">{t.get("trend_age_days","")}d trend</span>
            </div>
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
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        body, .gradio-container { 
            background: linear-gradient(135deg, #09090b, #111116) !important; 
            color: #ececec !important; 
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* Glassmorphism sidebar */
        .sidebar { 
            background: rgba(255, 255, 255, 0.03) !important; 
            backdrop-filter: blur(16px); 
            -webkit-backdrop-filter: blur(16px);
            padding: 24px; 
            border-radius: 20px; 
            border: 1px solid rgba(255, 255, 255, 0.05); 
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, background 0.3s ease;
        }
        .sidebar:hover { background: rgba(255, 255, 255, 0.05) !important; }
        
        .main-content { background: transparent !important; padding: 10px; }
        
        /* Cards */
        .card { 
            background: rgba(20, 20, 25, 0.5) !important; 
            backdrop-filter: blur(12px);
            padding: 24px; 
            border-radius: 18px; 
            border: 1px solid rgba(255, 255, 255, 0.08); 
            margin-bottom: 20px; 
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease-in-out;
        }
        .card:hover {
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }
        
        /* Buttons */
        .spotify-btn { 
            background: linear-gradient(90deg, #1DB954, #1ed760) !important; 
            color: #000 !important; 
            font-weight: 800 !important; 
            border-radius: 50px !important; 
            border: none !important;
            box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3) !important;
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .spotify-btn:hover { 
            transform: translateY(-2px) scale(1.02) !important; 
            box-shadow: 0 6px 20px rgba(29, 185, 84, 0.5) !important; 
        }
        .spotify-btn:active { transform: scale(0.98) !important; }
        
        .secondary-btn { 
            background: rgba(255, 255, 255, 0.05) !important; 
            color: #ececec !important; 
            font-weight: 600 !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important; 
            border-radius: 50px !important; 
            transition: all 0.2s ease !important;
        }
        .secondary-btn:hover { 
            background: rgba(255, 255, 255, 0.1) !important; 
            border-color: rgba(255, 255, 255, 0.3) !important;
            transform: scale(1.03) !important; 
        }
        
        /* Tables */
        table { 
            width: 100%; border-collapse: collapse; 
            background: rgba(0, 0, 0, 0.3); 
            border-radius: 12px; overflow: hidden; 
            backdrop-filter: blur(5px);
        }
        th { 
            background: rgba(255, 255, 255, 0.03); 
            color: #a0a0a0; text-align: left; 
            padding: 16px 12px; font-size: 0.85em; 
            text-transform: uppercase; letter-spacing: 1px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        td { 
            border-bottom: 1px solid rgba(255, 255, 255, 0.03); 
            padding: 14px 12px; color: #ececec; font-size: 0.95em; 
            transition: background 0.2s;
        }
        tr:hover td { background: rgba(255, 255, 255, 0.08); cursor: pointer; }
        tr:last-child td { border-bottom: none; }
        
        /* Typography */
        h1 { 
            color: #fff !important; 
            font-weight: 800 !important;
            letter-spacing: -1px;
            background: -webkit-linear-gradient(#fff, #b3b3b3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2, h3 { color: #fff !important; font-weight: 600 !important; margin-bottom: 10px !important; }
        .muted { color: #9ca3af !important; font-size: 0.95em; line-height: 1.5; }
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
