import contextlib
import asyncio
import requests
import os
import io
import gradio as gr
import re
import random

from modules import scripts, shared, script_callbacks
from scripts.Gel import Gelbooru
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

def _expand_or_pattern_simple(s: str) -> str:
    if not s:
        return s
    pattern = re.compile(r"\(([^()]+)\)")
    def repl(match):
        parts = match.group(1).split("|")
        parts = [p.strip() for p in parts if p.strip()]
        return random.choice(parts) if parts else ""
    return pattern.sub(repl, s)


# ==========================================================
# Utility: async-safe runner
# ==========================================================
def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# ==========================================================
# Removal list: file-backed helpers (extensions-local)
#   - File: extensions/Gelbooru-Prompt-Randomizer/removal_tags.txt
#   - UTF-8 / 1行1タグ / 空行OK / 先頭が#の行はコメント
#   - 比較は：trim -> lower -> " "→"_" の正規化で一致判定
# ==========================================================
_REMOVAL_CACHE = {"mtime": None, "set": set()}

def _removal_file_path() -> str:
    ext_root = os.path.dirname(os.path.dirname(__file__))
    list_dir = os.path.join(ext_root, "list")
    os.makedirs(list_dir, exist_ok=True)
    return os.path.join(list_dir, "removal_tags.txt")

def _normalize_tag(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def _parse_removal_text_to_set(text: str) -> set:
    out = set()
    if not text:
        return out
    text = text.replace("\r", "\n")
    for line in text.split("\n"):
        if not line:
            continue
        if line.lstrip().startswith("#"):
            continue
        parts = [p for p in line.split(",")] if "," in line else [line]
        for p in parts:
            t = _normalize_tag(p)
            if t:
                out.add(t)
    return out

def _read_removal_text() -> str:
    path = _removal_file_path()
    if not os.path.exists(path):
        return ""
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _write_removal_text(content: str) -> None:
    path = _removal_file_path()
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(content if content is not None else "")
    os.replace(tmp, path)

def _load_removal_set(force: bool = False) -> set:
    path = _removal_file_path()
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    if not force and _REMOVAL_CACHE["mtime"] == mtime:
        return _REMOVAL_CACHE["set"]
    text = _read_removal_text()
    s = _parse_removal_text_to_set(text)
    _REMOVAL_CACHE["mtime"] = mtime
    _REMOVAL_CACHE["set"] = s
    return s

def _apply_removal_filter(raw_tags: list) -> list:
    """
    Gelbooruの生タグ列（lower/underscore想定）に除外集合を適用。
    失敗しても生成は止めない。
    """
    try:
        removal = _load_removal_set(force=False)
        if not removal:
            return raw_tags
        return [t for t in raw_tags if t and _normalize_tag(t) not in removal]
    except Exception:
        return raw_tags

# ---------- UI glue for Removal List ----------
def _ui_load_removal_text() -> str:
    return _read_removal_text()

def _ui_reload_removal_text():
    _load_removal_set(force=True)
    return _read_removal_text()

def _ui_save_removal_text(content: str):
    _write_removal_text(content or "")
    _load_removal_set(force=True)
    return content or ""

# ==========================================================
# Core tag fetcher (shared between UI & auto mode)
# ==========================================================
async def get_random_tags(include, exclude):
    include = _expand_or_pattern_simple(include)
    exclude = _expand_or_pattern_simple(exclude)
    # 既存仕様：スペースはすべて削除、カンマ区切り
    include = include.replace(" ", "")
    exclude = exclude.replace(" ", "")
    api_key = getattr(shared.opts, "gpr_api_key", None)
    user_id = getattr(shared.opts, "gpr_user_id", None)

    if(api_key == "" or user_id == ""):
        return "You need to log in to your gelbooru account", None, "You need to log in to your gelbooru account"

    include = include.split(',') if include else None
    exclude = exclude.split(',') if exclude else None

    gel_post = await Gelbooru(api_key=api_key, user_id=user_id).random_post(tags=include, exclude_tags=exclude)
    if(gel_post == None or gel_post == []):
        return "Couldn't find a post with the specified tags", None, "Couldn't find a post with the specified tags"
    
    tags = gel_post.get_tags()
    # ★ 追加：出力時の除外フィルタ（TXT）
    tags = _apply_removal_filter(tags)

    # 既存仕様：除外リストに無いものだけ "_"→" " の可読化
    excl = getattr(shared.opts, "gpr_undersocreReplacementExclusionList").split(',')
    for i in range(len(tags)):
        if(tags[i] not in excl):
            tags[i] = tags[i].replace("_", " ")

    # --- 安全化: 画像URLの存在確認 ---
    image_url = getattr(gel_post, "file_url", None)
    if not image_url or not isinstance(image_url, str) or not image_url.strip():
        image_url = None
    else:
        try:
            resp = requests.head(image_url, timeout=5)
            if resp.status_code != 200:
                image_url = None  # 死んだURLはスキップ
        except Exception:
            image_url = None  # 接続失敗も同様にスキップ

    return ', '.join(tags), image_url, str(gel_post)




def _fetch_tags_sync(include_str, exclude_str):
    """Used in before_process (sync wrapper)."""
    api_key = getattr(shared.opts, "gpr_api_key", None)
    user_id = getattr(shared.opts, "gpr_user_id", None)
    if not api_key or not user_id:
        return None

    include_str = _expand_or_pattern_simple(include_str)
    exclude_str = _expand_or_pattern_simple(exclude_str)

    # 既存仕様：スペースはすべて削除、カンマ区切り
    include = include_str.replace(" ", "") if include_str else ""
    exclude = exclude_str.replace(" ", "") if exclude_str else ""
    include_list = include.split(',') if include else None
    exclude_list = exclude.split(',') if exclude else None

    gel = Gelbooru(api_key=api_key, user_id=user_id)
    post = _run_async(gel.random_post(tags=include_list, exclude_tags=exclude_list))
    if not post:
        return None

    tags = post.get_tags()
    # ★ 追加：出力時の除外フィルタ（TXT）
    tags = _apply_removal_filter(tags)

    # 既存仕様：除外リストに無いものだけ "_"→" " の可読化
    excl = getattr(shared.opts, "gpr_undersocreReplacementExclusionList").split(',')
    processed = [t.replace("_", " ") if t not in excl else t for t in tags]
    return processed


# ==========================================================
# Main UI Script
# ==========================================================
class GPRScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enable_auto = None
        self.include_box = None
        self.exclude_box = None

    def title(self):
        return "Gelbooru Prompt Randomizer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Gelbooru Prompt Randomizer', open=False):
            with gr.Column():
                self.include_box = gr.Textbox(label='Include Tags', placeholder="e.g. 1girl, blue_hair, solo")
                self.exclude_box = gr.Textbox(label='Exclude Tags', placeholder="e.g. nsfw, text, watermark")
                self.enable_auto = gr.Checkbox(label="Enable Auto Mode (fetch before generation)", value=False)

                # ----- Removal List (TXT-backed) -----
                with gr.Group():
                    removal_textbox = gr.Textbox(
                        label="Removal List (comma-separated or newline-separated tags; lines starting with # are comments)",
                        value=_ui_load_removal_text(),
                        lines=3
                    )
                    with gr.Row():
                        removal_save_btn = gr.Button(value='Save Removal List', variant='primary', size='sm')
                        removal_reload_btn = gr.Button(value='Reload', size='sm')

                with gr.Row():
                    send_text_button = gr.Button(value='Randomize', variant='primary', size='sm')
                    append_tags_button = gr.Button(value='Append Tags', size='sm')
                    clear_button = gr.Button(value='Clear', size='sm')

                result_tags_textbox = gr.Textbox(label='Tags', show_copy_button=True, interactive=False)
                preview_image = gr.Image(interactive=False, show_label=False, height=400)
                url_textbox = gr.Textbox(label='Post URL', show_copy_button=True, interactive=False)

        # === Button bindings ===
        with contextlib.suppress(AttributeError):
            # Removal Save/Reload
            removal_save_btn.click(
                fn=_ui_save_removal_text,
                inputs=[removal_textbox],
                outputs=[removal_textbox],
            )
            removal_reload_btn.click(
                fn=_ui_reload_removal_text,
                inputs=None,
                outputs=[removal_textbox],
            )

            append_tags_button.click(
                fn=lambda result_tags, tags: (f"{tags}, {result_tags}"),
                inputs=[result_tags_textbox, self.text2img if not is_img2img else self.img2img],
                outputs=self.text2img if not is_img2img else self.img2img,
            )
            send_text_button.click(
                fn=get_random_tags,
                inputs=[self.include_box, self.exclude_box],
                outputs=[result_tags_textbox, preview_image, url_textbox],
            )
            clear_button.click(
                fn=lambda: (None, None, None),
                inputs=None,
                outputs=[preview_image, url_textbox, result_tags_textbox],
            )

        # Return order: must match before_process args
        return [self.enable_auto, self.include_box, self.exclude_box]

    # ======================================================
    # 自動実行フック：EnableがONのときだけ実行
    # ======================================================
    def before_process(self, p, enable_auto, include_box, exclude_box):
        if not enable_auto:
            return

        try:
            # 1回のリクエストでタグ＋画像URL＋Post情報取得
            tags_str, image_url, post_info = _run_async(get_random_tags(include_box, exclude_box))
        except Exception as e:
            print("[GPR] get_random_tags failed:", e)
            return

        # エラーメッセージ系は無視
        if not tags_str or "You need" in tags_str or "Couldn't find" in tags_str:
            return

        # ---- プロンプトにタグを追加 ----
        if getattr(p, "prompt", ""):
            p.prompt = f"{p.prompt}, {tags_str}"
        else:
            p.prompt = tags_str

        # ----------------------------------------------------
        # SDXL 推奨解像度への自動調整（縮小のみ・比率最適化）
        # ----------------------------------------------------
        def _find_best_sdxl_size(w, h):
            presets = [
                (1024, 1024),
                (1152, 896), (1216, 832), (1344, 768),
                (1536, 640), (1568, 672), (1728, 576),
                (896, 1152), (832, 1216), (768, 1344),
                (640, 1536), (576, 1728), (512, 2048),
            ]

            aspect = w / h
            best = None
            best_diff = 999.0

            for pw, ph in presets:
                # ソースより大きいサイズはスキップ（縮小のみ）
                if pw > w or ph > h:
                    continue

                diff = abs(aspect - (pw / ph))
                if diff < best_diff:
                    best_diff = diff
                    best = (pw, ph)

            return best

        # ---- img2img のときのみ画像を投入＆解像度最適化 ----
        if isinstance(p, StableDiffusionProcessingImg2Img) and image_url:
            try:
                resp = requests.get(image_url, timeout=10)

                # Pillow で実体読み込み（ここで壊れた画像だと例外）
                img = Image.open(io.BytesIO(resp.content))
                img.load()
                img = img.convert("RGB")

                # ソース画像の実サイズから、最適な SDXL 推奨解像度を決定
                px = img.width * img.height
                if px <= 1_100_000:  # 1.1M以下は拡大も縮小も禁止
                    print(f"[GPR] Keep original size due to low pixel count: {img.width}x{img.height}")
                    p.width, p.height = img.width, img.height
                else:
                    best = _find_best_sdxl_size(img.width, img.height)
                    if best:
                        p.width, p.height = best
                        print(f"[GPR] Auto SDXL Resize → {p.width}x{p.height}")
                    else:
                        print(f"[GPR] No suitable SDXL preset for {img.width}x{img.height}, keep original")

                p.init_images = [img]
                print(f"[GPR] Init Image Loaded (source) = {img.width}x{img.height}")

            except Exception as e:
                print("[GPR] Invalid image. Skip init image:", e)
                return  # ←ここが最重要

        # Include Tags をメタデータに保存
        try:
            include_raw = include_box if include_box else ""
            if include_raw:
                if not hasattr(p, "extra_generation_params"):
                    p.extra_generation_params = {}
                p.extra_generation_params["GPR Include Tags"] = include_raw
        except Exception as e:
            print("[GPR] Failed to insert include tags metadata:", e)

        # メタデータに投稿ページURLを保存
        try:
            if post_info:
                post_url = str(post_info)
                if not hasattr(p, "extra_generation_params"):
                    p.extra_generation_params = {}
                p.extra_generation_params["GPR Post URL"] = post_url
        except Exception as e:
            print("[GPR] Metadata insert failed:", e)

    # ======================================================
    # 設定UI（既存）
    # ======================================================
    def on_ui_settings():
        GPR_SECTION = ("gpr", "Gelbooru Prompt Randomizer")

        gpr_options = {
            "gpr_api_key": shared.OptionInfo("", "api_key", gr.Textbox).info("<a href=\"https://gelbooru.com/index.php?page=account&s=options\" target=\"_blank\">Account Options</a>"),
            "gpr_user_id": shared.OptionInfo("", "user_id", gr.Textbox).info("<a href=\"https://gelbooru.com/index.php?page=account&s=options\" target=\"_blank\">Account Options</a>"),
            "gpr_replaceUnderscores": shared.OptionInfo(True, "Replace underscores with spaces on insertion"),
            "gpr_undersocreReplacementExclusionList": shared.OptionInfo(
                "0_0,(o)_(o),+_+,+_-,._.,<o>_<o>,<|>_<|>,=_=,>_<,3_3,6_9,>_o,@_@,^_^,o_o,u_u,x_x,|_|,||_||",
                "Underscore replacement exclusion list"
            ).info("Add tags that shouldn't have underscores replaced with spaces, separated by comma."),
        }

        for key, opt in gpr_options.items():
            opt.section = GPR_SECTION
            shared.opts.add_option(key, opt)

    script_callbacks.on_ui_settings(on_ui_settings)

    # capture prompt components for Append button
    def after_component(self, component, **kwargs):
        if kwargs.get("elem_id") == "txt2img_prompt":
            self.text2img = component
        if kwargs.get("elem_id") == "img2img_prompt":
            self.img2img = component
