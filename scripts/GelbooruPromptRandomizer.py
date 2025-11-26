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

used_post_ids = set()
total_count = None

def _expand_or_pattern_simple(s: str) -> str:
    if not s:
        return s    
    pattern = re.compile(r"\{([^{}]*\|[^{}]*)\}")
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

def _apply_removal_filter(raw_tags: list, removal_text: str = "") -> list:
    try:  
        removal = _parse_removal_text_to_set(removal_text)
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

def register_used_post(post_info):
    global used_post_ids, total_count
    if not post_info:
        return

    pid = getattr(post_info, "id", None)
    if pid:
        pid = str(pid)
        if pid in used_post_ids:
            print(f"[GPR] Duplicate detected: {pid} (skip)")
        used_post_ids.add(pid)

    # 総数取得
    tc = getattr(post_info, "total_count", None)
    if tc:
        total_count = int(tc)

def is_all_used():
    if total_count is None:
        return False
    return len(used_post_ids) >= total_count

def _fetch_tags_sync(include_str, exclude_str, removal_text=""):
    include_raw = _expand_or_pattern_simple(include_str or "")
    exclude_raw = _expand_or_pattern_simple(exclude_str or "")
    include_raw = include_raw.replace(" ", "")
    exclude_raw = exclude_raw.replace(" ", "")
    include = include_raw.split(",") if include_raw else []
    exclude = exclude_raw.split(",") if exclude_raw else []

    try:
        gel = Gelbooru(shared.opts.gpr_api_key, shared.opts.gpr_user_id)
        results = _run_async(gel.search_posts(tags=include, exclude_tags=exclude, limit=1000))

        if not results:
            print("[GPR] No results from API")
            return "", None, None

        # 使用済み・DL失敗・NGタグを除外しながら次候補探索
        for gel_post in results:
            post_url = str(gel_post)
            pid_match = re.search(r'id=(\d+)', post_url)
            if not pid_match:
                continue
            pid = pid_match.group(1)

            # 既に使用済みならスキップ
            if pid in used_post_ids:
                continue

            image_url = getattr(gel_post, "file_url", None)
            if not image_url:
                continue

            # DLテスト
            try:
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
            except Exception:
                print(f"[GPR] Skip invalid image: {image_url}")
                continue

            # NGタグ除外
            tags_list = getattr(gel_post, "tags", [])
            tags_str = ", ".join(tags_list)

            used_post_ids.add(pid)  # 使用済み管理（セッションのみ）
            return tags_str, image_url, post_url

        print("[GPR] No unused valid images left")
        return "", None, None

    except Exception as e:
        print("[GPR] _fetch_tags_sync failed:", e)
        return "", None, None


# ==========================================================
# Main UI Script
# ==========================================================
class GPRScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enable_auto = None
        self.include_box = None
        self.exclude_box = None
        self._last_include = ""
        self._last_exclude = ""
        self._cached_tags_str = ""
        self._cached_image_url = ""
        self._cached_post_info = None

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
                    self.removal_textbox = gr.Textbox(
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
                inputs=[self.removal_textbox],
                outputs=[self.removal_textbox],
            )
            removal_reload_btn.click(
                fn=_ui_reload_removal_text,
                inputs=None,
                outputs=[self.removal_textbox],
            )

            append_tags_button.click(
                    fn=lambda result_tags, tags, rm: (
                    ", ".join(_apply_removal_filter(result_tags.split(", "), rm))
                    if result_tags else ""
                ),
                inputs=[result_tags_textbox, self.text2img if not is_img2img else self.img2img, self.removal_textbox],
                outputs=self.text2img if not is_img2img else self.img2img,
            )

            send_text_button.click(
                fn=lambda inc, exc, rm: _fetch_tags_sync(inc, exc, rm),
                inputs=[self.include_box, self.exclude_box, self.removal_textbox],
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

        # ==== タグ条件変更検知して使用済みリセット ====
        cur_inc = include_box or ""
        cur_exc = exclude_box or ""
        if cur_inc != self._last_include or cur_exc != self._last_exclude:
            used_post_ids.clear()
            self._cached_tags_str = ""
            self._cached_image_url = ""
            self._cached_post_info = None
            print("[GPR] Tag conditions changed → used list cleared")
        self._last_include = cur_inc
        self._last_exclude = cur_exc

        try:
            # === 必ず1回だけfetch ===
            if not self._cached_post_info:
                tags_str, image_url, post_info = _fetch_tags_sync(include_box, exclude_box)
                self._cached_tags_str = tags_str
                self._cached_image_url = image_url
                self._cached_post_info = post_info
            else:
                tags_str = self._cached_tags_str
                image_url = self._cached_image_url
                post_info = self._cached_post_info

        except Exception as e:
            print("[GPR] before_process failed (fetch_tags):", e)
            return

        if not tags_str or not image_url:
            print("[GPR] before_process: no valid tags/image → Auto stop")
            print(f"[GPR] Used: {len(used_post_ids)} / Cycle max 100")
            try:
                p.batch_count = 1
            except:
                pass
            return

        # ---- プロンプトにタグを追加 ----      
        rm = self.removal_textbox.value
        filtered = ", ".join(_apply_removal_filter(tags_str.split(", "), rm))
        if getattr(p, "prompt", ""):
            p.prompt = f"{p.prompt}, {filtered}"
        else:
            p.prompt = filtered

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
                if px <= 300_000:  # 指定値以下は拡大も縮小も禁止。推薦値は1_100_000（=1.1M）
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

                # ---------------------------------------------
                # 検証用：実際に参照したソース画像を保存（POST ID名）
                # ---------------------------------------------
                try:
                    import re
                    # POST ID 抽出
                    pid_match = re.search(r'id=(\d+)', str(post_info))
                    if pid_match:
                        pid = pid_match.group(1)
        
                        # 保存先ディレクトリ
                        import os
                        ext_root = os.path.dirname(os.path.dirname(__file__))
                        src_dir = os.path.join(ext_root, "source_images")
                        os.makedirs(src_dir, exist_ok=True)
        
                        # JPEG形式で保存
                        save_path = os.path.join(src_dir, f"{pid}.jpg")
                        img.save(save_path, format="JPEG")
                        print(f"[GPR] Source image saved → {save_path}")
                    else:
                        print("[GPR] No post ID found → source image not saved")
                except Exception as e:
                    print("[GPR] Failed to save source image:", e)
                # ---------------------------------------------
                # 検証用：実際に参照したソース画像を保存　ここまで
                # ---------------------------------------------

            except Exception as e:  
                print("[GPR] Invalid image. Trying next candidate:", e)

                # === NG画像も使用済みとして登録 ===
                try:
                    pid_match = re.search(r'id=(\d+)', str(post_info))
                    if pid_match:
                        used_post_ids.add(pid_match.group(1))
                except:
                    pass

                # === 次サイクルで必ず新しい候補を取得させる ===
                self._cached_post_info = None
                self._cached_image_url = ""
                self._cached_tags_str = ""

                return  # ←今回はこの生成をスキップ。次ループで再取得

        # Include Tags をメタデータに保存
        try:
            include_raw = include_box if include_box else ""
            if include_raw:
                if not hasattr(p, "extra_generation_params"):
                    p.extra_generation_params = {}
                p.extra_generation_params["GPR Include Tags"] = include_raw.replace('"', '')
        except Exception as e:
            print("[GPR] Failed to insert include tags metadata:", e)

        # メタデータに投稿ページURLを保存
        try:
            if post_info:

                # --- POST ページ URL（引用符除去） ---
                post_url = str(self._cached_post_info).replace('"', '')

                if not hasattr(p, "extra_generation_params"):
                    p.extra_generation_params = {}

                p.extra_generation_params["GPR Post URL"] = post_url

                # --- 画像取得用 file_url の併記（引用符除去） ---
                img_url = (self._cached_image_url or "").replace('"', '')
                p.extra_generation_params["GPR Image URL"] = img_url
        
                # -----------------------------------------
                # 末尾：使用済みURL登録（前方から移動）
                # -----------------------------------------
                try:
                    register_used_post(post_info)
                except Exception as e:
                    print("[GPR] register_used_post failed:", e)

                # -----------------------------------------
                # Auto stop （前方から移動）
                # -----------------------------------------
                if is_all_used():
                    try:
                        p.batch_count = 1
                        print(f"[GPR] All {total_count} posts used. Auto stop next cycle.")
                    except:
                        pass

                # -----------------------------------------
                # 条件付きキャッシュクリア
                # 「使用済み登録されなかったURLのみ保持」
                #   → register_used_post が成功していれば削除する
                # -----------------------------------------
                if self._cached_image_url:
                    pid_match = re.search(r'id=(\d+)', str(post_info))
                    pid = pid_match.group(1) if pid_match else None
                    if pid and pid in used_post_ids:
                        # normally clear
                        self._cached_tags_str = ""
                        self._cached_image_url = ""
                        self._cached_post_info = None
                        print("[GPR] Cache cleared (registered)")
                    else:
                        # keep cache because URL not registered
                        print("[GPR] Cache kept (unregistered)")

    
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
