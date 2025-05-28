import os
import time
from playsound import playsound
from collections import defaultdict

# 定義語音檔案的儲存資料夾名稱
AUDIO_CACHE_DIR = "audio_cache"

# 獲取當前腳本的絕對路徑
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 組合語音檔案的完整儲存路徑
AUDIO_BASE_PATH = os.path.join(current_script_dir, AUDIO_CACHE_DIR)

# 確保語音儲存資料夾存在，如果不存在則創建 (雖然現在是手動放置，但作為防禦性程式碼保留)
os.makedirs(AUDIO_BASE_PATH, exist_ok=True)

# 定義語音類型與對應預設音訊檔案名稱的映射
# 這裡的鍵 (key) 應該與你在 risk_analyzer.py 和 drowsiness_detection_mediapipe.py 中
# 呼叫 generate_and_play_audio 時傳入的 'alert_type' 參數完全一致
PRESET_AUDIO_MAP = {
    "risk_side_alert": "yellow.mp3",
    "risk_high_alert": "red.mp3",
    "drowsiness_alert": "tierd.mp3",
    # 如果你有其他語音提示，可以在這裡新增映射
    # "yawn_alert": "yawn_alert.mp3",
    # "test_one": "test_one.mp3", # 如果你有用於測試的固定音檔
    # "test_two": "test_two.mp3", # 如果你有用於測試的固定音檔
}

# 儲存每個警報類型上次播放的時間
_last_played_alert_time = defaultdict(float)

def generate_and_play_audio(text, alert_type, cooldown_seconds=5, lang='zh-tw'):
    """
    播放預設的語音檔案。
    包含冷卻時間機制，避免頻繁播放。

    Args:
        text (str): (此參數在此模式下不再用於生成語音，但保留其用於 API 一致性)
                    要轉換為語音的文字。
        alert_type (str): 警報的類型，用於從 PRESET_AUDIO_MAP 中查找對應的音訊檔案。
        cooldown_seconds (int): 該類型警報的冷卻時間 (秒)。
        lang (str): (此參數在此模式下不再用於生成語音，但保留其用於 API 一致性) 語音的語言代碼。
    Returns:
        bool: 語音是否成功播放。
    """
    current_time = time.time()

    # 檢查是否在冷卻時間內
    if current_time - _last_played_alert_time[alert_type] < cooldown_seconds:
        # print(f"Alert type '{alert_type}' is in cooldown. Skipping audio.")
        return False

    # 根據 alert_type 查找對應的預設音訊檔案名稱
    audio_filename = PRESET_AUDIO_MAP.get(alert_type)

    if not audio_filename:
        print(f"Error: No preset audio file found for alert type '{alert_type}'.")
        return False

    # 組合完整的音訊檔案路徑
    filepath = os.path.join(AUDIO_BASE_PATH, audio_filename)

    # 檢查預設音訊檔案是否存在
    if not os.path.exists(filepath):
        print(f"Error: Preset audio file not found at '{filepath}'. Please ensure it exists.")
        return False

    try:
        print(f"Playing preset audio: {filepath}")
        playsound(filepath, block=False) # 非阻塞播放
        _last_played_alert_time[alert_type] = current_time # 更新上次播放時間
        return True
    except Exception as e:
        print(f"Error playing sound {filepath}: {e}")
        return False

# 範例用法 (你可以在其他文件中呼叫 generate_and_play_audio 函式)
if __name__ == "__main__":
    print("This is speech_alert_system.py. Run this file directly for testing.")

    # 測試語音 (請確保這些測試類型在 PRESET_AUDIO_MAP 中有對應的檔案)
    print("Testing 'risk_side_alert'...")
    generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=2)
    time.sleep(1)
    generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=2) # 應該不會播放，因為在冷卻時間內
    time.sleep(2)
    print("Testing 'drowsiness_alert'...")
    generate_and_play_audio("你是來開車還是來睡覺的", "drowsiness_alert", cooldown_seconds=2)
    time.sleep(1)
    generate_and_play_audio("你是來開車還是來睡覺的", "drowsiness_alert", cooldown_seconds=2) # 應該不會播放，因為在冷卻時間內
    time.sleep(2)
    print("Testing 'risk_side_alert' again (should play now)...")
    generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=2) # 應該會播放，因為冷卻時間已過