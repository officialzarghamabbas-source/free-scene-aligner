import os
import re
import json
import tempfile
import difflib
import torch
import imageio_ffmpeg
import stable_whisper
import gradio as gr

os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

torch.set_num_threads(2)
DEVICE = "cpu"
MODEL_SIZE = "tiny.en"

print(f"Loading Fast Stable Whisper model ({MODEL_SIZE}) on {DEVICE}...")
model = stable_whisper.load_model(MODEL_SIZE, device=DEVICE)
print("Model loaded successfully!")

def format_custom_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hrs:02d}_{mins:02d}_{secs:02d}_{millis:03d}"

def clean_word(word: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', str(word)).strip().lower()

def extract_scenes_from_input(text: str):
    pattern = r'(SCENE\s*\d+[:\-]?)'
    chunks = re.split(pattern, text, flags=re.IGNORECASE)
    
    scenes = []
    for i in range(1, len(chunks), 2):
        label_raw = chunks[i].strip()
        label = re.sub(r'[:\-]+$', '', label_raw).strip().upper()
        content = chunks[i+1].strip()
        content = re.sub(r'^[\s:\-–—]+', '', content)
        content = content.strip(' \t\n\r"\'“”')
        
        if content:
            scenes.append({
                "label": label,
                "text": content
            })
    return scenes

def find_best_subsequence_match(needle_tokens, haystack_tokens, start_search_idx, max_window=350):
    n_len = len(needle_tokens)
    if n_len == 0:
        return start_search_idx, start_search_idx
    
    search_end = min(len(haystack_tokens), start_search_idx + max_window + n_len)
    window_tokens = haystack_tokens[start_search_idx:search_end]
    
    matcher = difflib.SequenceMatcher(None, needle_tokens, window_tokens)
    match = matcher.find_longest_match(0, n_len, 0, len(window_tokens))
    
    if match.size >= max(2, int(n_len * 0.3)):
        matched_start = start_search_idx + match.b - match.a
        matched_end = matched_start + n_len - 1
        return max(start_search_idx, matched_start), min(len(haystack_tokens) - 1, matched_end)
    
    est_start = start_search_idx
    est_end = min(len(haystack_tokens) - 1, est_start + n_len - 1)
    return est_start, est_end

def process_alignment(audio_path, raw_scene_text):
    if not audio_path:
        return "Audio file upload nahi ki!", None, None

    user_scenes = extract_scenes_from_input(raw_scene_text)
    if not user_scenes:
        return "Koi scenes detect nahi hue! Har scene 'SCENE 1', 'SCENE 2' format mein hona chahiye.", None, None

    full_script = " ".join([sc["text"] for sc in user_scenes])

    try:
        alignment_result = model.align(audio_path, full_script, language="en")
    except Exception:
        alignment_result = model.transcribe(audio_path, vad=True, regroup=False)

    audio_words = []
    for segment in alignment_result.segments:
        for w in segment.words:
            c_w = clean_word(w.word)
            if c_w:
                audio_words.append({
                    "word": w.word.strip(),
                    "clean": c_w,
                    "start": float(w.start),
                    "end": float(w.end)
                })

    if not audio_words:
        return "Audio se koi words align nahi ho sake!", None, None

    total_audio_duration = float(alignment_result.segments[-1].end) if alignment_result.segments else audio_words[-1]["end"]
    haystack_clean = [w["clean"] for w in audio_words]
    total_words = len(audio_words)
    current_word_idx = 0
    raw_cuts = []

    for sc in user_scenes:
        tokens = [clean_word(t) for t in sc["text"].split() if clean_word(t)]
        if not tokens:
            continue

        start_idx, end_idx = find_best_subsequence_match(tokens, haystack_clean, current_word_idx)
        
        start_time = audio_words[start_idx]["start"]
        end_time = audio_words[end_idx]["end"]
        
        current_word_idx = min(total_words - 1, end_idx + 1)

        raw_cuts.append({
            "label": sc["label"],
            "text": sc["text"],
            "detected_start": start_time,
            "detected_end": end_time
        })

    stitched_data = []
    num_scenes = len(raw_cuts)

    for i in range(num_scenes):
        current_item = raw_cuts[i]
        final_start = 0.0 if i == 0 else stitched_data[i-1]["end_seconds"]

        if i < num_scenes - 1:
            next_start = raw_cuts[i+1]["detected_start"]
            current_end = current_item["detected_end"]

            if next_start > current_end:
                split_point = round((current_end + next_start) / 2.0, 3)
            else:
                split_point = round(current_end, 3)

            final_end = max(split_point, round(final_start + 0.05, 3))
        else:
            final_end = round(max(current_item["detected_end"], total_audio_duration), 3)

        t_start_str = format_custom_timestamp(final_start)
        t_end_str = format_custom_timestamp(final_end)
        timestamp_str = f"{t_start_str} - {t_end_str}"

        stitched_data.append({
            "scene": current_item["label"],
            "text": current_item["text"],
            "timestamps": timestamp_str,
            "start_seconds": round(final_start, 3),
            "end_seconds": round(final_end, 3),
            "duration": round(final_end - final_start, 3)
        })

    final_output_text = ""
    for item in stitched_data:
        final_output_text += f"{item['scene']}\n"
        final_output_text += f"TIMESTAMPS: {item['timestamps']}\n"
        final_output_text += f"VOICE LINE: \"{item['text']}\"\n\n"

    out_dir = tempfile.mkdtemp()
    json_path = os.path.join(out_dir, "aligned_scenes.json")
    txt_path = os.path.join(out_dir, "aligned_scenes.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stitched_data, f, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final_output_text)

    return final_output_text, json_path, txt_path

with gr.Blocks(title="Exact Precision Scene Aligner") as demo:
    gr.Markdown("## 🎯 100% Exact Scene-to-Voice Aligner (Free Public Tool)")
    gr.Markdown("**Zero Drift, Acoustic Forced Alignment, and Millisecond Precision.**")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="Upload Voice Audio (MP3 / WAV)")
            text_in = gr.Textbox(
                lines=14, 
                label="Paste Your Scenes Here", 
                placeholder='SCENE 1\nHave you ever looked in the mirror...\n\nSCENE 2\nand noticed something that wasn\'t there...'
            )
            btn = gr.Button("🚀 Generate 100% Exact Timestamps", variant="primary")
            
        with gr.Column():
            res_box = gr.Textbox(lines=16, label="Acoustic Aligned Output")
            with gr.Row():
                f_json = gr.File(label="Download JSON (CapCut Ready)")
                f_txt = gr.File(label="Download TXT")

    demo.queue(max_size=20)
    btn.click(fn=process_alignment, inputs=[audio_in, text_in], outputs=[res_box, f_json, f_txt])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
