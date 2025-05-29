import os
import time
import simpleaudio as sa
from collections import defaultdict
import threading
import queue
import gc 
import sys 
import traceback # 確保導入 traceback

# --- 語音系統配置區 ---
# 定義語音檔案的儲存資料夾名稱
AUDIO_CACHE_DIR = "audio_cache"

# 獲取當前腳本的絕對路徑，確保跨平台相容性
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 組合語音檔案的完整儲存路徑
AUDIO_BASE_PATH = os.path.join(current_script_dir, AUDIO_CACHE_DIR)

# 定義語音類型與對應預設音訊檔案名稱的映射
PRESET_AUDIO_MAP = {
    "risk_side_alert": "risk_side_alert.wav",
    "risk_high_alert": "risk_high_alert.wav",
    "drowsiness_alert": "drowsiness_alert.wav",
}

# --- 新增：語音功能開關 ---
# 設置為 False 將完全禁用所有語音播放請求和後台線程
ENABLE_AUDIO_ALERTS = True 

# --- 內部狀態管理區 ---
_last_played_alert_finished_time = defaultdict(float)
_audio_queue = queue.Queue(maxsize=5) 
_is_playing_any_audio = False
_play_state_lock = threading.Lock()

# 全局變量來保存播放線程，只有在 ENABLE_AUDIO_ALERTS 為 True 時才啟動
_player_thread = None

# --- 背景播放線程功能區 ---
def _audio_player_worker():
    # ... (此函數內容與上次優化版本相同，保持不變) ...
    """
    負責從佇列中取出音頻並播放的工人執行緒。
    該線程會確保任何時候只有一個音頻在播放 (無重疊)。
    並在播放完成後更新冷卻時間。
    """
    global _is_playing_any_audio
    while True:
        wave_obj = None 
        play_obj = None 
        try:
            try:
                filepath, alert_type, cooldown_seconds_for_this_playback = _audio_queue.get(timeout=1)
                print(f"[Audio Worker Debug] Fetched request: {alert_type}, file: {filepath}")
            except queue.Empty:
                continue 

            with _play_state_lock:
                current_time = time.time()
                if current_time - _last_played_alert_finished_time[alert_type] < cooldown_seconds_for_this_playback:
                    print(f"[Audio Worker] Alert type '{alert_type}' is in cooldown. Skipping playback from queue.")
                    _audio_queue.task_done()
                    continue

                while _is_playing_any_audio:
                    time.sleep(0.01)

                _is_playing_any_audio = True 
                print(f"[Audio Worker] Attempting to play: {filepath} (Type: {alert_type})")

            try:
                wave_obj = sa.WaveObject.from_wave_file(filepath)
                print(f"[Audio Worker Debug] WaveObject loaded for {alert_type}.")
                play_obj = wave_obj.play()
                print(f"[Audio Worker Debug] Playback started for {alert_type}.")
                play_obj.wait_done() 
                print(f"[Audio Worker] Finished playing: {alert_type}")

            except sa.SimpleaudioError as sa_e:
                print(f"[Audio Worker Error] Simpleaudio playback error for {filepath}: {sa_e}")
                traceback.print_exc(file=sys.stdout)
            except FileNotFoundError:
                print(f"[Audio Worker Error] Audio file not found during playback: {filepath}")
            except Exception as playback_e:
                print(f"[Audio Worker Error] Unexpected error during audio playback of {filepath}: {playback_e}")
                traceback.print_exc(file=sys.stdout)
            finally:
                with _play_state_lock:
                    _last_played_alert_finished_time[alert_type] = time.time() 
                    _is_playing_any_audio = False 

                if play_obj is not None:
                    del play_obj
                if wave_obj is not None:
                    del wave_obj
                gc.collect() 

                _audio_queue.task_done() 

        except Exception as e:
            print(f"[Audio Worker Fatal Error] Unhandled exception in player worker, restarting loop: {e}")
            traceback.print_exc(file=sys.stdout)
            with _play_state_lock:
                _is_playing_any_audio = False
            pass

# --- 外部接口功能區 ---
def generate_and_play_audio(text, alert_type, cooldown_seconds=5, lang='zh-tw'):
    """
    外部調用接口：將語音播放請求加入內部佇列。
    此函數會根據警報類型和當前播放狀態來決定是否將請求排隊。
    """
    global _player_thread # 聲明使用全局變量

    # 如果音頻警報被禁用，則直接返回
    if not ENABLE_AUDIO_ALERTS:
        return False

    # 如果線程還未啟動，則在此處啟動
    if _player_thread is None:
        _player_thread = threading.Thread(target=_audio_player_worker, daemon=True)
        _player_thread.start()
        print("[Speech Alert System] Audio player worker thread started.") # 首次啟動提示

    filepath = os.path.join(AUDIO_BASE_PATH, PRESET_AUDIO_MAP.get(alert_type))

    if not os.path.exists(filepath):
        print(f"Error: Preset audio file not found at '{filepath}'. Please ensure it exists.")
        return False

    with _play_state_lock: 
        current_time = time.time()

        if current_time - _last_played_alert_finished_time[alert_type] < cooldown_seconds:
            print(f"[{alert_type}] is in cooldown. Skipping request.")
            return False

        if _audio_queue.full():
            print(f"[Queue Full] Audio queue is full. Skipping request for '{alert_type}'.")
            return False

        if _is_playing_any_audio and alert_type != "risk_high_alert":
            print(f"[Audio Busy] Another audio is currently playing. Skipping request for '{alert_type}'.")
            return False

    _audio_queue.put((filepath, alert_type, cooldown_seconds))
    print(f"[Request Added] Audio request for '{alert_type}' added to queue.")
    return True

# --- 模組測試區 (僅在此檔案直接運行時執行) ---
if __name__ == "__main__":
    print("This is speech_alert_system.py. Run this file directly for testing.")

    # ... (測試代碼與上次相同，保持不變) ...
    print("\n--- Testing 'risk_high_alert' (should repeat after finish, no overlap) ---")
    generate_and_play_audio("已進入危險範圍，請立即減速", "risk_high_alert", cooldown_seconds=1.5)
    time.sleep(0.5)
    generate_and_play_audio("已進入危險範圍，請立即減速", "risk_high_alert", cooldown_seconds=1.5)
    time.sleep(0.5)
    generate_and_play_audio("已進入危險範圍，請立即減速", "risk_high_alert", cooldown_seconds=1.5)
    time.sleep(5)

    print("\n--- Testing 'risk_side_alert' (should play once, no overlap) ---")
    generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=5)
    time.sleep(1)
    generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=5)
    time.sleep(5)
    generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=5)
    time.sleep(3)

    print("\n--- Testing 'drowsiness_alert' (should play once, no overlap) ---")
    generate_and_play_audio("偵測到你疲勞了，請保持清醒或稍作休息", "drowsiness_alert", cooldown_seconds=5)
    time.sleep(1)
    generate_and_play_audio("偵測到你疲勞了，請保持清醒或稍作休息", "drowsiness_alert", cooldown_seconds=5)
    time.sleep(5)
    generate_and_play_audio("偵測到你疲勞了，請保持清醒或稍作休息", "drowsiness_alert", cooldown_seconds=5)
    time.sleep(3)

    print("\n--- Testing mixed alerts (should queue and play sequentially, respecting cooldowns) ---")
    generate_and_play_audio("高風險警報", "risk_high_alert", cooldown_seconds=1.5)
    time.sleep(0.1)
    generate_and_play_audio("側邊警報", "risk_side_alert", cooldown_seconds=5)
    time.sleep(0.1)
    generate_and_play_audio("疲勞警報", "drowsiness_alert", cooldown_seconds=5)
    time.sleep(2)
    generate_and_play_audio("高風險持續", "risk_high_alert", cooldown_seconds=1.5)
    time.sleep(10)
    print("Test finished. Please observe console output and audio playback.")