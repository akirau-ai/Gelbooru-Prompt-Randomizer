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
            # API取得時にPOSTページURLを確定させる
            raw_id = getattr(gel_post, "id", None)
            if raw_id is None:
                continue
            pid = str(raw_id)
            post_url = f"https://gelbooru.com/index.php?page=post&s=view&id={pid}"

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

        self._fatal_api_error = False

    def title(self):
        return "Gelbooru Prompt Randomizer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Gelbooru Prompt Randomizer', open=False):
            with gr.Column():

                # ---- 1段目：Include (単独) ----
                self.include_box = gr.Textbox(
                    label='Include Tags',
                    placeholder="e.g. 1girl, blue_hair, solo",
                )

                # ---- 2段目：Exclude + Scale（横並び）----
                with gr.Row():
                    self.exclude_box = gr.Textbox(
                        label='Exclude Tags',
                        placeholder='e.g. nsfw, text, watermark',
                        scale=3
                    )

                    self.scale_box = gr.Textbox(
                        label='SDXL Resize Scale',
                        value="1.0",
                        placeholder="1.0 / 1.2 / 1.5",
                        scale=1
                    )
                # ---- Enable Auto Mode ----
                self.enable_auto = gr.Checkbox(
                    label="Enable Auto Mode (fetch before generation)",
                    value=False
                )

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
        return [self.enable_auto, self.include_box, self.exclude_box, self.scale_box]

    # ======================================================
    # 自動実行フック：EnableがONのときだけ実行
    # ======================================================
    def before_process(self, p, enable_auto, include_box, exclude_box, scale_box):
        if not enable_auto:
            return

        # ==== タグ条件変更検知（このpに紐づくキャッシュだけ無効化）====
        cur_inc = include_box or ""
        cur_exc = exclude_box or ""

        # ----------------------------------------------
        # ★ 新サイクル開始時の fatal error リセット
        #   （＝ generate ボタン押下 → p が新しくなる）
        # ----------------------------------------------
        if self._fatal_api_error:
            print("[GPR] Reset fatal API error (new cycle)")
            self._fatal_api_error = False

        if cur_inc != self._last_include or cur_exc != self._last_exclude:
            print("[GPR] Tag conditions changed → cache cleared for this cycle")
            if hasattr(p, "_gpr_cached_post_info"):
                p._gpr_cached_post_info = None
                p._gpr_cached_image_url = None
                p._gpr_cached_tags_str = ""
    
            # ----------------------------------------------
            # ★ タグ条件が変わった時点でも fatal error をリセット
            # ----------------------------------------------
            if self._fatal_api_error:
                print("[GPR] Reset fatal API error (tag changed)")
                self._fatal_api_error = False

        self._last_include = cur_inc
        self._last_exclude = cur_exc

        if self._fatal_api_error:
            print("[GPR] Fatal API error → skip before_process")
            return

        try:
            # === 必ず1サイクルにつき1回だけfetch（p単位のキャッシュ） ===
            if not hasattr(p, "_gpr_cached_post_info") or not p._gpr_cached_post_info:
                tags_str, image_url, post_info = _fetch_tags_sync(include_box, exclude_box)
                p._gpr_cached_tags_str = tags_str
                p._gpr_cached_image_url = image_url
                p._gpr_cached_post_info = post_info
            else:
                tags_str = p._gpr_cached_tags_str
                image_url = p._gpr_cached_image_url
                post_info = p._gpr_cached_post_info

        except Exception as e:
            print("[GPR] before_process failed (fetch_tags):", e)
            return

        # --- API失敗時の再取得ループ（最大5回・2秒間隔） ---
        retry_count = 0
        while (not tags_str or not image_url) and retry_count < 5:
            retry_count += 1
            print(f"[GPR] API retry {retry_count}/5 (waiting 2s)")
            import time
            time.sleep(2)

            try:
                tags_str, image_url, post_info = _fetch_tags_sync(include_box, exclude_box)
                # リトライ成功時も p 側キャッシュを更新
                p._gpr_cached_tags_str = tags_str
                p._gpr_cached_image_url = image_url
                p._gpr_cached_post_info = post_info
            except Exception as e:
                print(f"[GPR] retry fetch failed: {e}")
                tags_str, image_url, post_info = "", None, None

        # --- 最終判定（10回失敗） ---
        if not tags_str or not image_url:
            print("[GPR] API failed after 5 retries → Auto stop.")
            print(f"[GPR] Used: {len(used_post_ids)} / Cycle max 100")

            # ★ 次サイクルの before_process を完全停止するフラグ
            self._fatal_api_error = True

            try:
                p.batch_count = 1
            except:
                pass
            return

        # --- リトライ成功時は p 側キャッシュ更新 ---
        p._gpr_cached_tags_str = tags_str
        p._gpr_cached_image_url = image_url
        p._gpr_cached_post_info = post_info

        # ---- プロンプトにタグを追加（filtered） ----
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
                # ---- SDXL Resize Scale ----
                try:
                    scale = float(scale_box)
                except:
                    scale = 1.0

                def _round64(x):
                    return max(64, int(round(x / 64)) * 64)

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
                        # --- apply scale & round to 64 ---
                        scaled_w = _round64(best[0] * scale)
                        scaled_h = _round64(best[1] * scale)
                        p.width, p.height = scaled_w, scaled_h
                        print(f"[GPR] Auto SDXL Resize × {scale} → {p.width}x{p.height}")

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

                # === このpに紐づくキャッシュを無効化して、次の before で再取得させる ===
                try:
                    p._gpr_cached_post_info = None
                    p._gpr_cached_image_url = None
                    p._gpr_cached_tags_str = ""
                except Exception:
                    pass

                return  # ←今回はこの生成をスキップ。次の before で再取得                    

        # ---- メタデータ初回書き込み ----
        try:
            include_raw = include_box if include_box else ""
            if not hasattr(p, "extra_generation_params"):
                p.extra_generation_params = {}

            # Include Tags（未設定時のみ）
            if include_raw and "GPR Include Tags" not in p.extra_generation_params:
                p.extra_generation_params["GPR Include Tags"] = include_raw.replace('"', '')
        except Exception as e:
            print("[GPR] Failed to insert include tags metadata:", e)

        # ---- POST URL をメタデータへ（未設定時のみ）----
        try:
            if post_info:
                post_url = str(post_info).replace('"', '')  

                if not hasattr(p, "extra_generation_params"):
                    p.extra_generation_params = {}

                # Post URL（未設定時のみ）
                if "GPR Post URL" not in p.extra_generation_params:
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
